import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import PIL.Image
import torch
import torch.multiprocessing
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from controlnet_aux.lineart import LineartDetector
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    T2IAdapter,
)

from one_shot.face import FaceHelper, FaceHelperModels
from one_shot.params import Params, Settings
from one_shot.prompt import Compels, Prompts
from one_shot.utils import (
    civitai_path,
    exclude,
)

if TYPE_CHECKING:
    from loguru._logger import Logger

    from one_shot.dreambooth.process import ProcessRequest


@dataclass
class SharedModels:
    detector: LineartDetector
    face: FaceHelperModels


@dataclass
class ProcessModels:
    pipe: StableDiffusionXLAdapterPipeline
    refiner: Optional[StableDiffusionXLImg2ImgPipeline]
    inpainter: StableDiffusionXLInpaintPipeline
    compels: Compels
    face: FaceHelperModels
    settings: Settings = Settings()

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
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            params.model.name,
            vae=AutoencoderKL.from_pretrained(
                params.model.vae,
                **exclude(cls.settings.loading_kwargs, {"variant"}),
            ).to(rank),
            adapter=T2IAdapter.from_pretrained(
                params.model.t2i_adapter,
                **exclude(cls.settings.loading_kwargs, {"variant"}),
                varient="fp16",
            ).to(rank),
            scheduler=EulerAncestralDiscreteScheduler.from_pretrained(
                params.model.name, subfolder="scheduler"
            ),
            **cls.settings.loading_kwargs,
        ).to(rank)
        pipe.enable_xformers_memory_efficient_attention()
        cls._load_loras(params, pipe)

        if params.use_refiner():
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                params.model.refiner,
                text_encoder_2=pipe.text_encoder_2,
                vae=pipe.vae,
                scheduler=pipe.scheduler,
                **cls.settings.loading_kwargs,
            ).to(rank)
            refiner.enable_xformers_memory_efficient_attention()
        else:
            refiner = None

        inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
            params.model.inpainter,
            text_encoder=pipe.text_encoder,
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            scheduler=pipe.scheduler,
            **cls.settings.loading_kwargs,
        ).to(rank)
        inpainter.enable_xformers_memory_efficient_attention()
        cls._load_loras(params, inpainter, key="inpainter")

        xl_compel = Compel(
            [pipe.tokenizer, pipe.tokenizer_2],
            [pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=rank,
            textual_inversion_manager=DiffusersTextualInversionManager(pipe),
        )
        if params.use_refiner():
            refiner_compel = Compel(
                tokenizer=refiner.tokenizer_2,
                text_encoder=refiner.text_encoder_2,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=True,
                device=rank,
                textual_inversion_manager=DiffusersTextualInversionManager(pipe),
            )
        else:
            refiner_compel = None
        inpainter_compel = Compel(
            [inpainter.tokenizer, inpainter.tokenizer_2],
            [inpainter.text_encoder, inpainter.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=rank,
            textual_inversion_manager=DiffusersTextualInversionManager(inpainter),
        )

        return cls(
            pipe=pipe,
            refiner=refiner,
            inpainter=inpainter,
            compels=Compels(
                xl=xl_compel, refiner=refiner_compel, inpainter=inpainter_compel
            ),
            face=FaceHelperModels.load(rank),
        )


@dataclass
class Model:
    params: Params
    rank: int
    models: ProcessModels
    logger: "Logger"
    settings: Settings = Settings()

    @torch.inference_mode()
    def run(self, request: "ProcessRequest") -> list[PIL.Image.Image]:
        prompts = Prompts(
            self.models.compels,
            self.rank,
            self.params.dtype,
            [
                self.params.prompt_template.format(prompt=p, **request.demographics)
                for p in request.generation.prompts
            ],
            [
                self.params.negative_prompt + ", " + color
                for color in random.choices(
                    self.params.negative_colors, k=len(request.generation.prompts)
                )
            ],
        )
        self.logger.info("Generating latents...")
        generator = torch.Generator(device=self.rank)
        if self.params.seed:
            generator.manual_seed(self.params.seed)
        latents = [
            self.models.pipe(
                image=img,
                generator=generator,
                num_inference_steps=self.params.steps,
                guidance_scale=self.params.guidance_scale,
                adapter_conditioning_scale=random.triangular(
                    *self.params.conditioning_strength
                ),
                adapter_conditioning_factor=self.params.conditioning_factor,
                denoising_end=self.params.high_noise_frac,
                output_type="latent" if self.params.use_refiner() else "pil",
                **prompts.kwargs_for_xl(idx),
            ).images[0]
            for idx, img in enumerate(request.generation.images)
        ]

        if self.params.use_refiner():
            self.logger.info("Refining latents...")
            images = [
                self.models.refiner(
                    image=latent,
                    generator=generator,
                    num_inference_steps=self.params.steps,
                    guidance_scale=self.params.guidance_scale,
                    denoising_start=self.params.high_noise_frac,
                    strength=self.params.refiner_strength,
                    **prompts.kwargs_for_refiner(idx),
                ).images[0]
                for idx, latent in enumerate(latents)
            ]
        else:
            images = latents

        self.logger.info("Touching up images...")
        masks, colors = map(
            list,
            zip(*FaceHelper(self.params, self.models.face, images).eye_masks()),
        )
        self.logger.info("Colors: {}", colors)
        prompts = replace(
            prompts,
            raw=[
                self.params.inpaint_prompt_template.format(
                    prompt=p, color=c, **request.demographics
                )
                for p, c in zip(request.generation.prompts, colors)
            ],
            negative=[self.params.negative_prompt] * len(request.generation.prompts),
        )
        return [
            self.models.inpainter(
                image=img,
                mask_image=masks[idx],
                strength=self.params.inpainting_strength,
                num_inference_steps=self.params.inpainting_steps,
                **prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]
