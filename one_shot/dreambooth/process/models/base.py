from abc import ABC
from dataclasses import dataclass, field
from typing import TypeVar

import torch
import torch.multiprocessing
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)

from one_shot.face import FaceHelperModels
from one_shot.params.base import Params
from one_shot.params.settings import Settings
from one_shot.prompt import Compels
from one_shot.utils import civitai_path, exclude

T = TypeVar("T")


@dataclass
class SharedModels:
    face: FaceHelperModels


@dataclass
class ProcessModels:
    base: StableDiffusionXLPipeline
    inpainter: StableDiffusionXLInpaintPipeline
    refiner: StableDiffusionXLImg2ImgPipeline
    refine_inpainter: StableDiffusionXLInpaintPipeline
    compels: Compels
    face: FaceHelperModels
    settings: Settings = field(default_factory=Settings)

    @classmethod
    def _load_loras(cls, params: Params, pipe, key: str = "base"):
        for repo, lora in params.model.loras[key].items():
            if lora == "civitai":
                path = civitai_path(repo)
                pipe.load_lora_weights(
                    str(path.parent),
                    weight_name=str(path.name),
                    **cls.settings.loading_kwargs,
                )
            else:
                pipe.load_lora_weights(
                    repo, weight_name=lora, **cls.settings.loading_kwargs
                )
        pipe.fuse_lora(lora_scale=params.lora_scale)

    @classmethod
    @torch.inference_mode()
    def load(cls, params: Params, rank: int):
        base = StableDiffusionXLPipeline.from_pretrained(
            params.model.name,
            vae=AutoencoderKL.from_pretrained(
                params.model.vae,
                **exclude(cls.settings.loading_kwargs, {"variant"}),
            ).to(rank),
            scheduler=EulerAncestralDiscreteScheduler.from_pretrained(
                params.model.name, subfolder="scheduler"
            ),
            **cls.settings.loading_kwargs,
        ).to(rank)
        base.enable_xformers_memory_efficient_attention()
        cls._load_loras(params, base)

        inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
            params.model.inpainter,
            text_encoder=base.text_encoder,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            scheduler=base.scheduler,
            **cls.settings.loading_kwargs,
        ).to(rank, params.dtype)

        # inpainter.unet = torch.compile(
        #     inpainter.unet, mode="reduce-overhead", fullgraph=True
        # )
        inpainter.enable_xformers_memory_efficient_attention()
        cls._load_loras(params, inpainter, key="inpainter")

        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            params.model.refiner,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            scheduler=base.scheduler,
            **cls.settings.loading_kwargs,
        ).to(rank)
        # refiner.unet = torch.compile(
        #     refiner.unet, mode="reduce-overhead", fullgraph=True
        # )
        refiner.enable_xformers_memory_efficient_attention()

        refine_inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
            params.model.refiner,
            **refiner.components,
            **cls.settings.loading_kwargs,
        ).to(rank)
        refine_inpainter.enable_xformers_memory_efficient_attention()

        xl_compel = Compel(
            [base.tokenizer, base.tokenizer_2],
            [base.text_encoder, base.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=rank,
        )
        inpainter_compel = Compel(
            [inpainter.tokenizer, inpainter.tokenizer_2],
            [inpainter.text_encoder, inpainter.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=rank,
        )
        refiner_compel = Compel(
            tokenizer=refiner.tokenizer_2,
            text_encoder=refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            device=rank,
        )
        return cls(
            base=base,
            refiner=refiner,
            refine_inpainter=refine_inpainter,
            inpainter=inpainter,
            compels=Compels(
                xl=xl_compel,
                refiner=refiner_compel,
                inpainter=inpainter_compel,
            ),
            face=FaceHelperModels.load(params, rank),
        )

    @classmethod
    def load_child(cls, params: Params, klass: type[T], rank: int) -> T:
        models = cls.load(params, rank)
        return klass(models, params, rank)


@dataclass
class ProcessModelImpl(ABC):
    base: ProcessModels
