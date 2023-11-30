from typing import TYPE_CHECKING, Union

import torch
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.models import (
    AutoencoderKL,
    MultiAdapter,
    T2IAdapter,
    UNet2DConditionModel,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from loguru import logger
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

if TYPE_CHECKING:
    from one_shot.prompt import Prompts


class StableDiffusionXLAdapterEnsemblePipeline(StableDiffusionXLAdapterPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        adapter: Union[T2IAdapter, MultiAdapter, list[T2IAdapter]],
        refiner: StableDiffusionXLImg2ImgPipeline,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            adapter=adapter,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
        )
        self.register_modules(refiner=refiner)

    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image, list[Image.Image]],
        *args,
        prompts: "Prompts",
        w: float,
        refiner_strength: float,
        **kwargs,
    ):
        logger.info("Generating original latents")
        latents = (
            super()
            .__call__(
                image=image,
                *args,
                denoising_end=high_noise_frac,
                output_type="latent",
                **prompts.kwargs_for_xl(),
                **kwargs,
            )
            .images
        )
        logger.info("Refining latents")
        return self.refiner(
            *args,
            denoising_start=high_noise_frac,
            image=latents,
            **prompts.kwargs_for_refiner(),
            **{k: v for k, v in kwargs.items() if not k.startswith("adapter_")},
            strength=refiner_strength,
        )
