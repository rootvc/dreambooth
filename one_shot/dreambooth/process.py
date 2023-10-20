import itertools
import os
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from accelerate import PartialState
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from torch.multiprocessing import Queue

from one_shot.config import init_config
from one_shot.dreambooth.model import Model, ProcessModels
from one_shot.params import Params, Settings

if TYPE_CHECKING:
    from one_shot.dreambooth import Queues


T = TypeVar("T")


class GenerationRequest(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    images: list[Image.Image]
    prompts: list[str]
    params: dict[str, list[Any]] = Field(default_factory=dict)


class ProcessRequest(BaseModel):
    demographics: dict[str, str]
    generation: GenerationRequest


class ProcessResponseSentinel(BaseModel):
    rank: int = Field(default_factory=lambda: int(os.environ["RANK"]))


class ProcessResponse(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    images: list[Image.Image]
    params: dict[str, Any] = Field(default_factory=dict)
    rank: int = Field(default_factory=lambda: int(os.environ["RANK"]))


class Process:
    settings = Settings()

    @classmethod
    def run(
        cls,
        rank: int,
        world_size: int,
        params: Params,
        queues: "Queues",
    ):
        os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            store=dist.FileStore("/tmp/filestore", world_size),
        )
        init_config()
        proc = ProcessModels.load(params, rank)
        model = Model(
            params,
            rank,
            proc,
            logger,
        )
        cls(params, model, queues.proc[rank], queues.response).wait()

    def __init__(
        self,
        params: Params,
        model: Model,
        recv: "Queue[Optional[ProcessRequest]]",
        resp: "Queue[ProcessResponse | ProcessResponseSentinel]",
    ):
        self.params = params
        self.recv = recv
        self.resp = resp
        self.state = PartialState()
        assert self.state.device is not None
        self.logger = model.logger = logger.bind(rank=self.state.process_index)
        self.model = model

    def wait(self):
        self.logger.info("Waiting...")
        while True:
            if request := self.recv.get():
                self.logger.info("Received request: {}", request.generation.prompts)
                if request.generation.params:
                    self.tune(request)
                else:
                    self.generate(request)
            else:
                self.logger.info("Received stop signal")
                break

    @contextmanager
    def _split(self, obj: T) -> Iterator[T]:
        with self.state.split_between_processes(obj) as res:
            yield res

    def _pre_generate(self, request: ProcessRequest):
        params = [
            dict(zip(request.generation.params.keys(), v))
            for v in itertools.product(*request.generation.params.values())
        ]
        with self._split(params) as params:
            self.logger.debug("Params: {}", params)

        with self._split(request.generation.dict(exclude={"params"})) as split:
            generation = GenerationRequest(**split)
            self.logger.info("Slice: {}", generation)
        return generation, params

    def _generate(self, request: ProcessRequest, **params) -> list[Image.Image]:
        model = replace(self.model, params=self.model.params.copy(update=params))
        return model.run(request)

    @torch.inference_mode()
    def generate(self, request: ProcessRequest):
        generation, _ = self._pre_generate(request)
        self.logger.info("Generating images...")
        images = self._generate(request.copy(update={"generation": generation}))
        self.logger.debug("Sending response...")
        self.resp.put(ProcessResponse(images=images))
        self.resp.put(ProcessResponseSentinel())

    @torch.inference_mode()
    def tune(self, request: ProcessRequest):
        generation, paramsets = self._pre_generate(request)
        for params in paramsets:
            self.logger.info("Using params: {}", params)
            images = self._generate(
                request.copy(update={"generation": generation}), **params
            )
            self.resp.put(ProcessResponse(images=images, params=params))

        self.logger.info("Sending sentinel...")
        self.resp.put(ProcessResponseSentinel())