from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import PIL.Image
import snoop
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
    refiner: StableDiffusionXLImg2ImgPipeline
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
    @snoop
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

        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            params.model.refiner,
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            scheduler=pipe.scheduler,
            **cls.settings.loading_kwargs,
        ).to(rank)
        refiner.enable_xformers_memory_efficient_attention()

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
        refiner_compel = Compel(
            tokenizer=refiner.tokenizer_2,
            text_encoder=refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            device=rank,
            textual_inversion_manager=DiffusersTextualInversionManager(pipe),
        )
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
                self.params.negative_prompt  # + (
                # ", " + color
                # for color in random.choices(
                #     self.params.negative_colors, k=len(request.generation.prompts)
                # )
                # )
            ]
            * 2,
        )
        self.logger.info("Generating latents...")
        latents = [
            self.models.pipe(
                image=img,
                generator=torch.Generator(device=self.rank).manual_seed(42),
                num_inference_steps=25,
                guidance_scale=5.0,
                adapter_conditioning_scale=0.9,
                adapter_conditioning_factor=0.8,
                denoising_end=1.0,
                output_type="latent",
                **prompts.kwargs_for_xl(idx),
            ).images[0]
            for idx, img in enumerate(request.generation.images)
        ]

        self.logger.info("Refining latents...")
        images = [
            self.models.refiner(
                image=latent,
                generator=torch.Generator(device=self.rank).manual_seed(42),
                num_inference_steps=25,
                guidance_scale=5.0,
                denoising_start=1.0,
                strength=0.15,
                **prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, latent in enumerate(latents)
        ]

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
