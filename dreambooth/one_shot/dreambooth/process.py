import itertools
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from accelerate import PartialState
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
    T2IAdapter,
)
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from torch.multiprocessing import Queue
from torchvision.transforms.functional import to_tensor

from one_shot.config import init_config, init_tf
from one_shot.ensemble import StableDiffusionXLAdapterEnsemblePipeline
from one_shot.face import Face
from one_shot.params import Params, Settings
from one_shot.prompt import Compels, Prompts
from one_shot.utils import civitai_path

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


@dataclass
class ProcessModels:
    ensemble: StableDiffusionXLAdapterEnsemblePipeline
    inpainter: StableDiffusionXLInpaintPipeline
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
        init_config()
        init_tf()
        cls(params, queues.proc[rank], queues.response).wait()

    def __init__(
        self,
        params: Params,
        recv: "Queue[Optional[ProcessRequest]]",
        resp: "Queue[ProcessResponse | ProcessResponseSentinel]",
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
        torch.cuda.set_device(self.state.device)

        base: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
            self.params.model.name,
            vae=self.params.model.vae,
            **self.settings.loading_kwargs,
        ).to(self.state.device, torch_dtype=self.params.dtype)
        for repo, lora in self.params.model.loras["base"].items():
            if lora == "civitai":
                base.load_lora_weights(
                    str(civitai_path(repo)), **self.settings.loading_kwargs
                )
            else:
                base.load_lora_weights(
                    repo, weight_name=lora, **self.settings.loading_kwargs
                )
        base.fuse_lora(lora_scale=self.params.lora_scale)
        base = base.to(self.state.device, torch_dtype=self.params.dtype)
        base.enable_xformers_memory_efficient_attention()

        refiner: StableDiffusionXLImg2ImgPipeline = (
            StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.params.model.refiner,
                text_encoder_2=base.text_encoder_2,
                vae=base.vae,
                **self.settings.loading_kwargs,
            ).to(self.state.device, torch_dtype=self.params.dtype)
        )
        refiner.enable_xformers_memory_efficient_attention()
        refiner.scheduler = EulerAncestralDiscreteScheduler.from_config(
            refiner.scheduler.config
        )

        inpainter: StableDiffusionXLInpaintPipeline = (
            StableDiffusionXLInpaintPipeline.from_pretrained(
                self.params.model.inpainter,
                text_encoder=base.text_encoder,
                text_encoder_2=base.text_encoder_2,
                vae=base.vae,
                **self.settings.loading_kwargs,
            ).to(self.state.device, torch_dtype=self.params.dtype)
        )
        for repo, lora in self.params.model.loras["inpainter"].items():
            if lora == "civitai":
                path = civitai_path(repo)
                inpainter.load_lora_weights(
                    str(path.parent),
                    weight_name=str(path.name),
                    **self.settings.loading_kwargs,
                )
            else:
                inpainter.load_lora_weights(
                    repo, weight_name=lora, **self.settings.loading_kwargs
                )
        inpainter.fuse_lora(lora_scale=self.params.lora_scale)
        inpainter = inpainter.to(self.state.device, torch_dtype=self.params.dtype)
        inpainter.enable_xformers_memory_efficient_attention()

        adapter: T2IAdapter = T2IAdapter.from_pretrained(
            self.params.model.t2i_adapter, **self.settings.loading_kwargs
        )
        adapter.adapter = adapter.adapter.to(self.state.device, dtype=self.params.dtype)

        ensemble: StableDiffusionXLAdapterEnsemblePipeline = (
            StableDiffusionXLAdapterEnsemblePipeline.from_pretrained(
                self.params.model.name,
                text_encoder=base.text_encoder,
                text_encoder_2=base.text_encoder_2,
                vae=base.vae,
                unet=base.unet,
                refiner=refiner,
                adapter=adapter,
                **self.settings.loading_kwargs,
            ).to(self.state.device, torch_dtype=self.params.dtype)
        )
        ensemble.scheduler = EulerAncestralDiscreteScheduler.from_config(
            ensemble.scheduler.config
        )

        xl_compel = Compel(
            [ensemble.tokenizer, ensemble.tokenizer_2],
            [ensemble.text_encoder, ensemble.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=self.state.device,
            textual_inversion_manager=DiffusersTextualInversionManager(ensemble),
        )
        refiner_compel = Compel(
            tokenizer=ensemble.refiner.tokenizer_2,
            text_encoder=ensemble.refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            device=self.state.device,
            textual_inversion_manager=DiffusersTextualInversionManager(ensemble),
        )
        inpainter_compel = Compel(
            [inpainter.tokenizer, inpainter.tokenizer_2],
            [inpainter.text_encoder, inpainter.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=self.state.device,
            textual_inversion_manager=DiffusersTextualInversionManager(inpainter),
        )

        return ProcessModels(
            ensemble=ensemble,
            compels=Compels(
                xl=xl_compel, refiner=refiner_compel, inpainter=inpainter_compel
            ),
            inpainter=inpainter,
        )

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
            [
                self.params.negative_prompt + ", " + color
                for color in random.choices(
                    self.params.negative_colors, k=len(generation.prompts)
                )
            ],
        )
        return generation, params, images, prompts

    def _generate(
        self,
        generation: GenerationRequest,
        inputs: torch.Tensor,
        prompts: Prompts,
        demographics: dict[str, str],
        **params,
    ) -> list[Image.Image]:
        images = self.models.ensemble(
            **{
                "image": inputs,
                "prompts": prompts,
                "guidance_scale": self.params.guidance_scale,
                "adapter_conditioning_scale": random.triangular(
                    *self.params.conditioning_strength
                ),
                "adapter_conditioning_factor": self.params.conditioning_factor,
                "high_noise_frac": self.params.high_noise_frac,
                "num_inference_steps": self.params.steps,
                "refiner_strength": self.params.refiner_strength,
                **params,
            }
        ).images

        self.logger.info("Touching up images...")

        masks, colors = map(list, zip(*Face(self.params, images).eye_masks()))
        self.logger.info("Colors: {}", colors)

        prompts = replace(
            prompts,
            raw=[
                self.params.inpaint_prompt_template.format(
                    prompt=p, color=c, **demographics
                )
                for p, c in zip(generation.prompts, colors)
            ],
            negative=[self.params.negative_prompt] * len(generation.prompts),
        )

        return self.models.inpainter(
            **prompts.kwargs_for_inpainter(),
            image=images,
            mask_image=masks,
            strength=self.params.inpainting_strength,
            num_inference_steps=self.params.inpainting_steps,
        ).images

    @torch.inference_mode()
    def generate(self, request: ProcessRequest):
        generation, _, inputs, prompts = self._pre_generate(request)
        self.logger.info("Generating images...")
        images = self._generate(generation, inputs, prompts, request.demographics)
        self.logger.debug("Sending response...")
        self.resp.put(ProcessResponse(images=images))
        self.resp.put(ProcessResponseSentinel())

    @torch.inference_mode()
    def tune(self, request: ProcessRequest):
        generation, paramsets, inputs, prompts = self._pre_generate(request)
        for params in paramsets:
            self.logger.info("Using params: {}", params)
            images = self._generate(
                generation, inputs, prompts, request.demographics, **params
            )
            self.resp.put(ProcessResponse(images=images, params=params))

        self.logger.info("Sending sentinel...")
        self.resp.put(ProcessResponseSentinel())
