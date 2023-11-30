from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing
from controlnet_aux.lineart import LineartDetector
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)

from one_shot.dreambooth.process.models.base import ProcessModelImpl
from one_shot.params.t2i_adapter import Params
from one_shot.utils import (
    exclude,
)

if TYPE_CHECKING:
    from one_shot.dreambooth.process.models.base import (
        ProcessModels as BaseProcessModels,
    )


@dataclass
class ProcessModels(ProcessModelImpl):
    base: "BaseProcessModels"
    detector: LineartDetector
    pipe: StableDiffusionXLAdapterPipeline

    @classmethod
    @torch.inference_mode()
    def load(cls, models: "BaseProcessModels", params: Params, rank: int):
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            params.model.name,
            vae=AutoencoderKL.from_pretrained(
                params.model.vae,
                **exclude(models.settings.loading_kwargs, {"variant"}),
            ).to(rank),
            adapter=T2IAdapter.from_pretrained(
                params.model.t2i_adapter,
                **models.settings.loading_kwargs,
            ).to(rank),
            scheduler=EulerAncestralDiscreteScheduler.from_pretrained(
                params.model.name, subfolder="scheduler"
            ),
            **models.settings.loading_kwargs,
        ).to(rank)
        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.enable_xformers_memory_efficient_attention()
        models._load_loras(params, pipe)

        detector = LineartDetector.from_pretrained(
            params.model.detector,
        ).to(f"cuda:{torch.cuda.device_count() - 1}")

        return cls(base=models, detector=detector, pipe=pipe)
