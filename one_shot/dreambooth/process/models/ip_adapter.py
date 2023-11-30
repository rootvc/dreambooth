from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
)
from ip_adapter import IPAdapterFull as _IPAdapterFull
from PIL import Image

from one_shot.dreambooth.process.models.base import ProcessModelImpl
from one_shot.params.ip_adapter import Params
from one_shot.utils import (
    exclude,
    load_hf_file,
)

if TYPE_CHECKING:
    from one_shot.dreambooth.process.models.base import (
        ProcessModels as BaseProcessModels,
    )


class IPAdapterFull(_IPAdapterFull):
    def generate(
        self,
        image: Image.Image,
        scale: float,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        num_inference_steps: int = 30,
        **kwargs,
    ):
        image = image.resize((224, 224))

        self.set_scale(scale)
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(
            bs_embed * num_samples, seq_len, -1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            1, num_samples, 1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
            bs_embed * num_samples, seq_len, -1
        )

        prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat(
            [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
        )

        return self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images


@dataclass
class ProcessModels(ProcessModelImpl):
    base: "BaseProcessModels"
    pipe: StableDiffusionPipeline
    ip_adapter: IPAdapterFull

    @classmethod
    @torch.inference_mode()
    def load(cls, models: "BaseProcessModels", params: Params, rank: int):
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(
            params.model.ip_adapter.vae,
            **exclude(models.settings.loading_kwargs, {"variant"}),
        ).to(rank)
        pipe = StableDiffusionPipeline.from_pretrained(
            params.model.ip_adapter.base,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            **exclude(models.settings.loading_kwargs, {"variant"}),
        ).to(rank)
        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.enable_xformers_memory_efficient_attention()

        ip_adapter = IPAdapterFull(
            pipe,
            Path(
                load_hf_file(
                    params.model.ip_adapter.repo,
                    params.model.ip_adapter.files.image_encoder,
                )
            ).parent,
            load_hf_file(
                params.model.ip_adapter.repo, params.model.ip_adapter.files.adapter
            ),
            rank,
            num_tokens=257,
        )

        compel = Compel(
            pipe.tokenizer,
            pipe.text_encoder,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED,
            requires_pooled=False,
            device=rank,
        )

        return cls(
            base=replace(models, compels=replace(models.compels, base=compel)),
            pipe=pipe,
            ip_adapter=ip_adapter,
        )
