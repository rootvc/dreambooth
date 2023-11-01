import random
from dataclasses import dataclass
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
    StableDiffusionXLPipeline,
    T2IAdapter,
)
from PIL import Image
from torchvision.transforms.functional import (
    to_pil_image,
)

from one_shot.face import Bounds, FaceHelper, FaceHelperModels
from one_shot.params import Params, PromptStrings, Settings
from one_shot.prompt import Compels, Prompts
from one_shot.utils import (
    Face,
    civitai_path,
    collect,
    dilate_mask,
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
    base: StableDiffusionXLPipeline
    pipe: StableDiffusionXLAdapterPipeline
    inpainter: StableDiffusionXLInpaintPipeline
    refiner: StableDiffusionXLInpaintPipeline
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

        base = StableDiffusionXLPipeline.from_pretrained(
            params.model.name,
            **exclude(pipe.components, {"adapter"}),
            **cls.settings.loading_kwargs,
        ).to(rank)
        base.enable_xformers_memory_efficient_attention()
        cls._load_loras(params, base)

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

        refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
            params.model.refiner,
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            scheduler=pipe.scheduler,
            **cls.settings.loading_kwargs,
        ).to(rank)
        refiner.enable_xformers_memory_efficient_attention()

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
            tokenizer=refiner.tokenizer_2,
            text_encoder=refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            device=rank,
            textual_inversion_manager=DiffusersTextualInversionManager(pipe),
        )
        return cls(
            base=base,
            pipe=pipe,
            refiner=refiner,
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

    @torch.inference_mode()
    def tune(self, request: "ProcessRequest") -> list[Image.Image]:
        instance = ModelInstance(self, request)
        outpainted = instance.outpaint()
        smooth_edges = instance._smooth_edges(outpainted)
        redo_background = instance._redo_background(smooth_edges)
        final_refine = instance._final_refine(redo_background)
        return [
            instance.backgrounds[0],
            instance.masks[0].convert("RGB"),
            instance.edge_masks[0].convert("RGB"),
            instance.outpaint_bases[0],
            #
            outpainted[0],
            smooth_edges[0],
            redo_background[0],
            final_refine[0],
        ]


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

    def _prompts(self, strings: PromptStrings, **kwargs) -> Prompts:
        return Prompts(
            self.models.compels,
            self.rank,
            self.params.dtype,
            [
                strings.positive(prompt=p, **self.request.demographics, **kwargs)
                for p in self.request.generation.prompts
            ],
            [
                strings.negative(prompt=p, **self.request.demographics, **kwargs)
                for p in self.request.generation.prompts
            ],
        )

    @cached_property
    def background_prompts(self) -> Prompts:
        return self._prompts(self.params.prompt_templates.background)

    def eyes_prompts(self, **kwargs) -> Prompts:
        return self._prompts(self.params.prompt_templates.eyes, **kwargs)

    @cached_property
    def merge_prompts(self) -> Prompts:
        return self._prompts(self.params.prompt_templates.merge)

    @cached_property
    def details_prompts(self) -> Prompts:
        return self._prompts(self.params.prompt_templates.details)

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
                **self.details_prompts.kwargs_for_xl(idx),
            ).images[0]
            for idx, img in enumerate(self.request.generation.images)
        ]

    def _touch_up_eyes(self, images: list[Image.Image]) -> list[Image.Image]:
        face_helper = FaceHelper(self.params, self.models.face, images)
        masks, colors = map(list, zip(*face_helper.eye_masks()))
        self.model.logger.info("Colors: {}", colors)
        return [
            self.models.inpainter(
                image=img,
                mask_image=masks[idx],
                generator=self.generator,
                strength=self.params.inpainting_strength,
                num_inference_steps=self.params.inpainting_steps,
                **self.eyes_prompts(color=colors[idx]).kwargs_for_inpainter(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]

    @cached_property
    def faces(self) -> list[Image.Image]:
        return self._touch_up_eyes(self._generate_face_images())

    @property
    def dims(self) -> tuple[int, int]:
        return (self.params.model.resolution, self.params.model.resolution)

    @cached_property
    @collect
    def frames(self) -> Generator[Image.Image, None, None]:
        for idx, face in enumerate(self.request.generation.faces):
            bounds = Bounds.from_face(self.dims, face)
            target_percent = random.triangular(0.525, 0.535)

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
            og_mask = ~erode_mask(face.mask.arr)
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
        for _, masks in self._outpaint_bases:
            yield masks

    @property
    def masks(self) -> list[Image.Image]:
        return list(map(itemgetter(0), self._masks))

    @property
    def og_masks(self) -> list[Image.Image]:
        return list(map(itemgetter(1), self._masks))

    @cached_property
    def edge_masks(self) -> list[Image.Image]:
        return [
            to_pil_image(
                (
                    dilate_mask(~np.asarray(self.masks[idx]))
                    - ~np.asarray(self.og_masks[idx])
                )
            )
            for idx in range(len(self._masks))
        ]

    def _base_background(self, idx: int) -> Image.Image:
        return self.models.base(
            generator=self.generator,
            guidance_scale=self.params.guidance_scale,
            num_inference_steps=self.params.inpainting_steps,
            **self.background_prompts.kwargs_for_xl(idx),
        ).images[0]

    @cached_property
    def backgrounds(self) -> list[Image.Image]:
        return [self._base_background(idx) for idx in range(len(self.frames))]

    @cached_property
    @collect
    def _outpaint_bases(
        self,
    ) -> Generator[tuple[Image.Image, tuple[Image.Image, Image.Image]], None, None]:
        bg_face_helper = FaceHelper(self.params, self.models.face, backgrounds)
        bg_face_bounds = bg_face_helper.primary_face_bounds()

        frame_face_helper = FaceHelper(self.params, self.models.face, self.frames)
        frame_face_bounds = frame_face_helper.primary_face_bounds()

        for idx, frame_img in enumerate(self.frames):
            bg = backgrounds[idx]
            bg_face = bg_face_bounds[idx][1]
            bg_bounds = Bounds.from_face(self.dims, bg_face)

            frame = np.asarray(frame_img)
            frame_face = frame_face_bounds[idx][1]
            frame_bounds = Bounds.from_face(self.dims, frame_face)
            frame_mask, frame_og_mask = self._get_masks(frame, frame_face)

            frame_mask = np.asarray(frame_mask.convert("RGB"))
            frame_og_mask = np.asarray(frame_og_mask.convert("RGB"))

            slice = (frame_mask == 0) & (frame != 0)
            masked_face = np.where(slice, frame, 0)
            masked_face = masked_face[frame_bounds.slice()]

            (Cx, Cy) = bg_bounds.center
            (Sx, Sy) = frame_bounds.size()
            start_x, start_y = max(Cx - Sx // 2, 0), max(Cy - Sy // 2, 0)
            end_x, end_y = min(Cx + Sx // 2, bg.width - 1), min(
                Cy + Sy // 2, bg.height - 1
            )

            if end_x - start_x != Sx:
                if start_x == 0:
                    end_x = Sx
                else:
                    start_x = end_x - Sx
            if end_y - start_y != Sy:
                if start_y == 0:
                    end_y = Sy
                else:
                    start_y = end_y - Sy

            bg_image = np.array(bg)
            bg_image[start_y:end_y, start_x:end_x] = masked_face

            mask = np.full((*self.dims, 3), 255, dtype=np.uint8)
            mask[start_y:end_y, start_x:end_x] = frame_mask[frame_bounds.slice()]
            og_mask = np.full((*self.dims, 3), 255, dtype=np.uint8)
            og_mask[start_y:end_y, start_x:end_x] = frame_og_mask[frame_bounds.slice()]

            yield to_pil_image(bg_image), (to_pil_image(mask), to_pil_image(og_mask))

    @property
    def outpaint_bases(self) -> list[Image.Image]:
        return list(map(itemgetter(0), self._outpaint_bases))

    @collect
    def outpaint(self) -> Generator[Image.Image, None, None]:
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
                **self.details_prompts.kwargs_for_inpainter(idx),
            ).images
            for idx, image in enumerate(self.outpaint_bases)
        ]

        refined = [
            self.models.refiner(
                image=latent,
                latents=latent,
                mask_image=self.og_masks[idx],
                generator=self.generator,
                strength=0.85,
                denoising_start=self.params.high_noise_frac,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.steps,
                **self.details_prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, latent in enumerate(latents)
        ]

        for idx, img in enumerate(refined):
            img = np.array(img)
            slice = np.asarray(self.masks[idx].convert("RGB")) == 0
            img[slice] = np.asarray(self.outpaint_bases[idx])[slice]
            yield to_pil_image(img)

    def _smooth_edges(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.refiner(
                image=img,
                mask_image=self.edge_masks[idx],
                generator=self.generator,
                strength=0.55,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.merge_prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]

    def _redo_background(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.inpainter(
                image=image,
                mask_image=self.masks[idx],
                generator=self.generator,
                strength=0.99,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.details_prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, image in enumerate(images)
        ]

    def _final_refine(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.refiner(
                image=img,
                mask_image=self.edge_masks[idx],
                generator=self.generator,
                strength=0.35,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.merge_prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]

    def generate(self) -> list[Image.Image]:
        outpainted = self.outpaint()
        smooth_edges = self._smooth_edges(outpainted)
        redo_background = self._redo_background(smooth_edges)
        return self._final_refine(redo_background)
