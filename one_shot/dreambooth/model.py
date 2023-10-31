import random
from dataclasses import dataclass, replace
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Generator

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
from torchvision.transforms.functional import (
    to_pil_image,
)

from one_shot.face import Bounds, FaceHelper, FaceHelperModels
from one_shot.params import Params, Settings
from one_shot.prompt import Compels, Prompts
from one_shot.utils import (
    Face,
    civitai_path,
    collect,
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
        return ModelInstance(self, request).generate()


@dataclass
class ModelInstance:
    model: Model
    request: "ProcessRequest"

    @property
    def rank(self):
        return self.model.rank

    @property
    def models(self):
        return self.model.models

    @property
    def params(self):
        return self.model.params

    @cached_property
    def prompts(self) -> Prompts:
        return Prompts(
            self.models.compels,
            self.rank,
            self.params.dtype,
            [
                (self.params.prompt_prefix + ", " + self.params.prompt_template).format(
                    prompt=p, **self.request.demographics
                )
                for p in self.request.generation.prompts
            ],
            [self.params.negative_prompt] * len(self.request.generation.prompts),
        )

    @cached_property
    def generator(self) -> torch.Generator | None:
        if self.model.params.seed:
            return torch.Generator(device=self.rank).manual_seed(self.params.seed)
        else:
            return None

    def _generate_face_images(self) -> list[Image.Image]:
        return [
            self.model.models.pipe(
                image=img,
                generator=self.generator,
                num_inference_steps=self.params.steps,
                guidance_scale=self.params.guidance_scale,
                adapter_conditioning_scale=random.triangular(
                    *self.params.conditioning_strength
                ),
                adapter_conditioning_factor=self.params.conditioning_factor,
                **self.prompts.kwargs_for_xl(idx),
            ).images[0]
            for idx, img in enumerate(self.request.generation.images)
        ]

    def _touch_up_face_images(self, images: list[Image.Image]) -> list[Image.Image]:
        face_helper = FaceHelper(self.params, self.models.face, images)
        masks, colors = map(list, zip(*face_helper.eye_masks()))
        self.model.logger.info("Colors: {}", colors)
        prompts = replace(
            self.prompts,
            raw=[
                self.params.inpaint_prompt_template.format(
                    prompt=p, color=c, **self.request.demographics
                )
                for p, c in zip(self.request.generation.prompts, colors)
            ],
        )
        return [
            self.models.inpainter(
                image=img,
                mask_image=masks[idx],
                generator=self.generator,
                strength=self.params.inpainting_strength,
                num_inference_steps=self.params.inpainting_steps,
                **prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]

    @cached_property
    def faces(self) -> list[Image.Image]:
        return self._touch_up_face_images(self._generate_face_images())

    @property
    def dims(self) -> tuple[int, int]:
        return (self.params.model.resolution, self.params.model.resolution)

    @cached_property
    @collect
    def frames(self) -> Generator[Image.Image, None, None]:
        for idx, face in enumerate(self.request.generation.faces):
            bounds = Bounds.from_face(self.dims, face)
            target_percent = random.triangular(0.58, 0.60)

            curr_width, curr_height = bounds.size()
            target_width, target_height = [int(x * target_percent) for x in self.dims]
            d_w, d_h = target_width - curr_width, target_height - curr_height
            delta = max(d_w, d_h)
            padding = (delta / max(self.dims)) / 2.0
            self.model.logger.info("Length Delta: {}, Padding: {}", delta, padding)

            slice = bounds.slice(padding)
            embed = np.asarray(self.faces[idx].resize(bounds.size(padding)))

            frame = np.zeros((*self.dims, 3), dtype=np.uint8)
            frame[slice] = embed
            yield to_pil_image(frame)

    def _get_masks(
        self, frame: np.ndarray, face: Face
    ) -> tuple[Image.Image, Image.Image]:
        if face.mask:
            self.model.logger.warning("Using face mask")
            mask = ~face.mask.arr
            og_mask = ~erode_mask(face.aggressive_mask.arr)
        elif not face.is_trivial:
            self.model.logger.warning("Using bounds mask")
            mask = np.full(frame.shape, 255, dtype=np.uint8)
            mask[
                Bounds.from_face(frame.shape[:2], face).slice(self.params.mask_padding)
            ] = 0
            og_mask = np.full(frame.shape, 255, dtype=np.uint8)
            og_mask[Bounds.from_face(frame.shape[:2], face).slice()] = 0
            og_mask = ~erode_mask(~og_mask)
        else:
            self.model.logger.warning("Using default mask")
            mask = np.full(frame.shape, 255, dtype=np.uint8)
            mask[frame != 0] = 0
            og_mask = ~erode_mask(~mask)

        return tuple(to_pil_image(m, mode="RGB").convert("L") for m in (mask, og_mask))

    @cached_property
    @collect
    def _masks(self) -> Generator[tuple[Image.Image, Image.Image], None, None]:
        face_helper = FaceHelper(self.params, self.models.face, self.frames)
        for idx, frame_img in enumerate(self.frames):
            frame = np.asarray(frame_img)
            face = face_helper.primary_face_bounds()[idx][1]
            yield self._get_masks(frame, face)

    @property
    def masks(self) -> list[Image.Image]:
        return list(map(itemgetter(0), self._masks))

    @property
    def og_masks(self) -> list[Image.Image]:
        return list(map(itemgetter(1), self._masks))

    @collect
    def _outpaint_bases(self) -> Generator[Image.Image, None, None]:
        for idx, frame_img in enumerate(self.frames):
            frame = np.asarray(frame_img)
            mask = self.masks[idx]
            image = np.array(
                self.faces[idx]
                .resize(frame.shape[:2])
                .filter(ImageFilter.BoxBlur(radius=100)),
                dtype=np.uint8,
            )
            idx = (np.asarray(mask.convert("RGB")) == 0) & (frame != 0)
            image[idx] = frame[idx]
            yield to_pil_image(image)

    @cached_property
    def outpaint_prompts(self) -> Prompts:
        return replace(
            self.prompts,
            raw=[
                (
                    self.params.refine_prompt_prefix
                    + ", "
                    + self.params.prompt_template
                ).format(prompt=p, **self.request.demographics)
                for p in self.request.generation.prompts
            ],
            negative=[
                self.params.refine_negative_prompt + ", " + self.params.negative_prompt
            ]
            * len(self.request.generation.prompts),
        )

    @collect
    def outpaint(self) -> Generator[Image.Image, None, None]:
        bases = self._outpaint_bases()

        latents = [
            self.models.inpainter(
                image=image,
                mask_image=self.masks[idx],
                generator=self.generator,
                strength=0.99,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.steps,
                output_type="latent",
                denoising_end=self.params.high_noise_frac,
                **self.outpaint_prompts.kwargs_for_inpainter(idx),
            ).images
            for idx, image in enumerate(bases)
        ]

        refined = [
            self.models.bg_refiner(
                image=latent,
                latents=latent,
                mask_image=self.og_masks[idx],
                generator=self.generator,
                strength=0.85,
                denoising_start=self.params.high_noise_frac,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.steps,
                **self.outpaint_prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, latent in enumerate(latents)
        ]

        for idx, img in enumerate(refined):
            img = np.array(img)
            slice = np.asarray(self.masks[idx].convert("RGB")) == 0
            img[slice] = np.asarray(bases[idx])[slice]
            yield to_pil_image(img)

    def generate(self) -> list[Image.Image]:
        outpainted = self.outpaint()
        edge_masks = [
            to_pil_image(
                (~np.asarray(self.masks[idx]) - ~np.asarray(self.og_masks[idx]))
            )
            for idx in range(len(outpainted))
        ]
        smooth_edges = [
            self.models.inpainter(
                image=img,
                mask_image=edge_masks[idx],
                generator=self.generator,
                strength=0.25,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, img in enumerate(outpainted)
        ]
        redo_background = [
            self.models.inpainter(
                image=img,
                mask_image=self.masks[idx],
                generator=self.generator,
                strength=0.75,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.outpaint_prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, img in enumerate(smooth_edges)
        ]
        return [
            self.models.bg_refiner(
                image=img,
                mask_image=edge_masks[idx],
                generator=self.generator,
                strength=0.25,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, img in enumerate(redo_background)
        ]
