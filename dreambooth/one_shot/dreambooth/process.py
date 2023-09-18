import os
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Optional, cast

import torch
import torch.distributed as dist
from accelerate import PartialState
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    T2IAdapter,
)
from loguru import logger
from PIL import Image
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
        resp: "Queue[list[Image.Image]]",
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

        refiner: StableDiffusionXLImg2ImgPipeline = (
            StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.params.model.refiner, **self.settings.loading_kwargs
            ).to(self.state.device, torch_dtype=self.params.dtype)
        )
        refiner.enable_xformers_memory_efficient_attention()

        inpainter: StableDiffusionXLInpaintPipeline = (
            StableDiffusionXLInpaintPipeline.from_pretrained(
                self.params.model.inpainter, **self.settings.loading_kwargs
            ).to(self.state.device, torch_dtype=self.params.dtype)
        )
        for repo, lora in self.params.model.loras["base"].items():
            if lora == "civitai":
                inpainter.load_lora_weights(
                    str(civitai_path(repo)), **self.settings.loading_kwargs
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
                refiner=refiner,
                adapter=adapter,
                **self.settings.loading_kwargs,
            ).to(self.state.device, torch_dtype=self.params.dtype)
        )

        for repo, lora in self.params.model.loras["base"].items():
            if lora == "civitai":
                ensemble.load_lora_weights(
                    str(civitai_path(repo)), **self.settings.loading_kwargs
                )
            else:
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
        inpainter_compel = Compel(
            [inpainter.tokenizer, inpainter.tokenizer_2],
            [inpainter.text_encoder, inpainter.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=self.state.device,
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

        images = self.models.ensemble(
            image=images,
            prompts=prompts,
            guidance_scale=self.params.guidance_scale,
            adapter_conditioning_scale=self.params.conditioning_strength,
            adapter_conditioning_factor=self.params.conditioning_factor,
            high_noise_frac=self.params.high_noise_frac,
            num_inference_steps=self.params.steps,
        ).images

        self.logger.info("Touching up images...")

        masks, colors = map(list, zip(*Face(self.params, images).eye_masks()))
        self.logger.info("Colors: {}", colors)

        prompts = replace(
            prompts,
            raw=[
                self.params.inpaint_prompt_template.format(
                    color=c, **request.demographics
                )
                for c in colors
            ],
            negative="",
        )

        images = self.models.inpainter(
            **prompts.kwargs_for_inpainter(),
            image=images,
            mask_image=masks,
            strength=self.params.inpainting_strength,
        ).images

        self.logger.info("Sending response...")
        self.resp.put(images)
