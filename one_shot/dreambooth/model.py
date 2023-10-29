import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.multiprocessing
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from controlnet_aux.lineart import LineartDetector
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLInpaintPipeline,
    T2IAdapter,
)
from PIL import Image, ImageFilter
from torchvision.transforms.functional import to_pil_image

from one_shot.face import Bounds, FaceHelper, FaceHelperModels
from one_shot.params import Params, Settings
from one_shot.prompt import Compels, Prompts
from one_shot.utils import (
    civitai_path,
    erode_mask,
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
    face_refiner: StableDiffusionXLAdapterPipeline
    bg_refiner: StableDiffusionXLInpaintPipeline
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

        face_refiner = StableDiffusionXLAdapterPipeline.from_pretrained(
            params.model.refiner,
            vae=pipe.vae,
            adapter=T2IAdapter.from_pretrained(
                params.model.t2i_adapter,
                **exclude(cls.settings.loading_kwargs, {"variant"}),
                varient="fp16",
            ).to(rank),
            **cls.settings.loading_kwargs,
        ).to(rank)
        face_refiner.enable_xformers_memory_efficient_attention()

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

        bg_refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
            params.model.refiner,
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            scheduler=pipe.scheduler,
            **cls.settings.loading_kwargs,
        ).to(rank)
        bg_refiner.enable_xformers_memory_efficient_attention()

        xl_compel = Compel(
            [pipe.tokenizer, pipe.tokenizer_2],
            [pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
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
        refiner_compel = Compel(
            tokenizer=bg_refiner.tokenizer_2,
            text_encoder=bg_refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            device=rank,
            textual_inversion_manager=DiffusersTextualInversionManager(pipe),
        )
        return cls(
            pipe=pipe,
            face_refiner=face_refiner,
            bg_refiner=bg_refiner,
            inpainter=inpainter,
            compels=Compels(
                xl=xl_compel, refiner=refiner_compel, inpainter=inpainter_compel
            ),
            face=FaceHelperModels.load(params, rank),
        )


@dataclass
class Model:
    params: Params
    rank: int
    models: ProcessModels
    logger: "Logger"
    settings: Settings = Settings()

    @torch.inference_mode()
    def run(self, request: "ProcessRequest") -> list[Image.Image]:
        prompts = og_prompts = Prompts(
            self.models.compels,
            self.rank,
            self.params.dtype,
            [
                (self.params.prompt_prefix + ", " + self.params.prompt_template).format(
                    prompt=p, **request.demographics
                )
                for p in request.generation.prompts
            ],
            [self.params.negative_prompt] * len(request.generation.prompts),
        )
        self.logger.info("Generating latents...")
        if self.params.seed:
            generator = torch.Generator(device=self.rank).manual_seed(self.params.seed)
        else:
            generator = None
        images = [
            self.models.pipe(
                image=img,
                generator=generator,
                num_inference_steps=self.params.steps,
                guidance_scale=self.params.guidance_scale,
                adapter_conditioning_scale=random.triangular(
                    *self.params.conditioning_strength
                ),
                adapter_conditioning_factor=self.params.conditioning_factor,
                **prompts.kwargs_for_xl(idx),
            ).images[0]
            for idx, img in enumerate(request.generation.images)
        ]

        self.logger.info("Touching up images...")
        face_helper = FaceHelper(self.params, self.models.face, images)
        masks, colors = map(list, zip(*face_helper.eye_masks()))
        self.logger.info("Colors: {}", colors)
        prompts = replace(
            og_prompts,
            raw=[
                self.params.inpaint_prompt_template.format(
                    prompt=p, color=c, **request.demographics
                )
                for p, c in zip(request.generation.prompts, colors)
            ],
        )
        faces: list[Image.Image] = [
            self.models.inpainter(
                image=img,
                mask_image=masks[idx],
                generator=generator,
                strength=self.params.inpainting_strength,
                num_inference_steps=self.params.inpainting_steps,
                **prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]

        self.logger.info("Reframing images...")
        frames: list[Image.Image] = []
        for idx, face in enumerate(request.generation.faces):
            dims = (self.params.model.resolution, self.params.model.resolution)
            padding = self.params.mask_padding * random.triangular(2.4, 2.8)

            bounds = Bounds.from_face(dims, face)
            slice = bounds.slice(padding)
            embed = np.asarray(faces[idx].resize(bounds.size(padding)))

            frame = np.zeros((*dims, 3), dtype=np.uint8)
            frame[slice] = embed
            frames.append(to_pil_image(frame))

        face_helper = FaceHelper(self.params, self.models.face, frames)

        self.logger.info("Outpainting images...")
        images, masks, og_masks = [], [], []
        for idx, frame_img in enumerate(frames):
            frame = np.asarray(frame_img)
            face = face_helper.primary_face_bounds()[idx][1]

            if face.mask:
                self.logger.warning("Using face mask")
                mask = ~face.mask.arr
                masks.append(to_pil_image(mask, mode="RGB").convert("L"))

                og_mask = ~erode_mask(face.mask.arr)
                og_masks.append(to_pil_image(og_mask, mode="RGB").convert("L"))
            elif not face.is_trivial:
                self.logger.warning("Using bounds mask")
                mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
                mask[
                    Bounds.from_face(frame.shape[:2], face).slice(
                        self.params.mask_padding
                    )
                ] = 0
                masks.append(to_pil_image(mask, mode="L"))

                og_mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
                og_mask[Bounds.from_face(frame.shape[:2], face).slice()] = 0
                og_masks.append(to_pil_image(og_mask, mode="L"))
            else:
                self.logger.warning("Using default mask")
                mask = np.full(frame.shape, 255, dtype=np.uint8)
                mask[frame != 0] = 0
                masks.append(to_pil_image(mask, mode="RGB").convert("L"))
                og_masks.append(to_pil_image(mask, mode="RGB").convert("L"))

            image = np.array(
                faces[idx]
                .resize(frame.shape[:2])
                .filter(ImageFilter.BoxBlur(radius=2)),
                dtype=np.uint8,
            )
            idx = (mask == 0) & (frame != 0)
            image[idx] = frame[idx]
            images.append(to_pil_image(image))

        latents = [
            self.models.inpainter(
                image=img,
                mask_image=mask,
                generator=generator,
                strength=0.99,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.steps,
                output_type="latent",
                **og_prompts.kwargs_for_inpainter(idx),
            ).images
            for img, mask, idx in zip(images, masks, range(len(images)))
        ]

        self.logger.info("Refining images...")
        prompts = replace(
            og_prompts,
            raw=[
                (
                    self.params.refine_prompt_prefix
                    + ", "
                    + self.params.prompt_template
                ).format(prompt=p, **request.demographics)
                for p in request.generation.prompts
            ],
            negative=[
                self.params.refine_negative_prompt + ", " + self.params.negative_prompt
            ]
            * len(request.generation.prompts),
        )
        refined = [
            self.models.bg_refiner(
                image=latent,
                mask_image=mask,
                generator=generator,
                strength=0.75,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.steps,
                **prompts.kwargs_for_refiner(idx),
            ).images[0]
            for (idx, latent), mask in zip(enumerate(latents), og_masks)
        ]

        latent_images = [
            self.models.inpainter.image_processor.postprocess(
                self.models.inpainter.vae.decode(
                    latent / self.models.inpainter.vae.config.scaling_factor,
                    return_dict=False,
                )[0]
            )[0]
            for latent in latents
        ]

        return [
            frames[0],
            images[0],
            masks[0].convert("RGB"),
            og_masks[0].convert("RGB"),
            latent_images[0],
            refined[0],
        ]
