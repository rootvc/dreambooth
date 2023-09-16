import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional, cast

import torch
import torch.distributed as dist
from accelerate import PartialState
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    T2IAdapter,
)
from loguru import logger
from PIL import Image
from torch.multiprocessing import SimpleQueue
from torchvision.transforms.functional import to_tensor

from one_shot.config import init_torch_config
from one_shot.ensemble import StableDiffusionXLAdapterEnsemblePipeline
from one_shot.params import Params, Settings
from one_shot.prompt import Compels, Prompts

if TYPE_CHECKING:
    from one_shot.dreambooth import Queues


@dataclass
class GenerationRequest:
    images: list[Image.Image]
    prompts: list[str]


@dataclass
class ProcessRequest:
    demographics: dict[str, str]
    generation: GenerationRequest


@dataclass
class ProcessModels:
    ensemble: StableDiffusionXLAdapterEnsemblePipeline
    compels: Compels


class Process:
    settings = Settings()

    @classmethod
    def run(cls, rank: int, world_size: int, params: Params, queues: "Queues"):
        os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            store=dist.FileStore("/tmp/filestore", world_size),
        )
        logger.info("Process {} started", rank)
        init_torch_config()
        cls(params, queues.proc[rank], queues.response).wait()

    def __init__(
        self,
        params: Params,
        recv: "SimpleQueue[Optional[ProcessRequest]]",
        resp: "SimpleQueue[list[Image.Image]]",
    ):
        self.params = params
        self.recv = recv
        self.resp = resp
        self.state = PartialState()
        assert self.state.device is not None
        self.logger = logger.bind(rank=self.state.process_index)
        self.models = self._load_models()

    @torch.inference_mode()
    def _load_models(self) -> ProcessModels:
        self.logger.info("Loading models...")

        refiner: StableDiffusionXLImg2ImgPipeline = (
            StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.params.model.refiner,
                safety_checker=None,
                **self.settings.loading_kwargs,
            ).to(self.state.device, torch_dtype=self.params.dtype)
        )
        refiner.enable_xformers_memory_efficient_attention()

        adapter: T2IAdapter = T2IAdapter.from_pretrained(
            self.params.model.t2i_adapter, **self.settings.loading_kwargs
        )
        adapter.adapter = adapter.adapter.to(self.state.device, dtype=self.params.dtype)

        ensemble: StableDiffusionXLAdapterEnsemblePipeline = (
            StableDiffusionXLAdapterEnsemblePipeline.from_pretrained(
                self.params.model.name,
                refiner=refiner,
                adapter=adapter,
                safety_checker=None,
                **self.settings.loading_kwargs,
            ).to(self.state.device, torch_dtype=self.params.dtype)
        )

        for repo, lora in self.params.model.loras.items():
            ensemble.load_lora_weights(
                repo, weight_name=lora, **self.settings.loading_kwargs
            )
        ensemble.fuse_lora(lora_scale=self.params.lora_scale)
        ensemble = ensemble.to(self.state.device, torch_dtype=self.params.dtype)

        ensemble.enable_xformers_memory_efficient_attention()
        ensemble.scheduler = EulerAncestralDiscreteScheduler.from_config(
            ensemble.scheduler.config
        )

        xl_compel = Compel(
            [ensemble.tokenizer, ensemble.tokenizer_2],
            [ensemble.text_encoder, ensemble.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=self.state.device,
        )
        refiner_compel = Compel(
            tokenizer=ensemble.refiner.tokenizer_2,
            text_encoder=ensemble.refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            device=self.state.device,
        )

        return ProcessModels(
            ensemble=ensemble, compels=Compels(xl=xl_compel, refiner=refiner_compel)
        )

    def wait(self):
        self.logger.info("Waiting...")
        while True:
            if request := self.recv.get():
                self.logger.info("Received request: {}", request.generation.prompts)
                self.generate(request)
            else:
                self.logger.info("Received stop signal")
                break

    @torch.inference_mode()
    def generate(self, request: ProcessRequest):
        with self.state.split_between_processes(asdict(request.generation)) as split:
            generation = GenerationRequest(**cast(dict, split))

        images = torch.stack([to_tensor(i) for i in generation.images]).to(
            self.state.device, dtype=self.params.dtype
        )
        prompts = Prompts(
            self.models.compels,
            self.state.device,
            self.params.dtype,
            [
                self.params.prompt_template.format(prompt=p, **request.demographics)
                for p in generation.prompts
            ],
            self.params.negative_prompt,
        )

        self.logger.warning(
            "Device: {}, {}", self.state.device, self.models.ensemble.device
        )

        images = self.models.ensemble(
            image=images,
            prompts=prompts,
            guidance_scale=self.params.guidance_scale,
            adapter_conditioning_scale=self.params.conditioning_strength,
            adapter_conditioning_factor=self.params.conditioning_factor,
            high_noise_frac=self.params.high_noise_frac,
            num_inference_steps=self.params.steps,
        ).images

        self.logger.info("Sending response...")
        self.resp.put(images)
