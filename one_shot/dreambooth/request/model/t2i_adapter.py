import random
from dataclasses import dataclass
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Generator

import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from one_shot.dreambooth.request.model.base import Model as BaseModel
from one_shot.dreambooth.request.model.base import ModelInstance as BaseModelInstance
from one_shot.face import Bounds
from one_shot.params.t2i_adapter import Params
from one_shot.utils import (
    collect,
    unsharp_mask,
)

if TYPE_CHECKING:
    from one_shot.dreambooth.process.models.t2i_adapter import ProcessModels
    from one_shot.dreambooth.process.process import ProcessRequest


@dataclass
class Model(BaseModel):
    params: Params
    models: "ProcessModels"

    def _tune(self, request: "ProcessRequest") -> list[Image.Image]:
        instance = ModelInstance(self, request)
        instance.outpaint_bases
        outpainted = instance.outpaint()
        smooth_edges = instance._smooth_edges(outpainted)
        redo_background = instance._redo_background(smooth_edges)
        final_refine = instance._final_refine(redo_background)
        return [
            # instance.request.generation.images[0],
            # instance.faces[0],
            # instance.frames[0],
            # instance.tmp[2].convert("RGB"),  # masked face
            # instance.controls[0].convert("RGB"),
            # instance.backgrounds[0],
            # ,
            instance.outpaint_bases[0],
            instance.masks[0].convert("RGB"),
            instance.mask_sms[0].convert("RGB"),
            instance.edge_masks[0].convert("RGB"),
            # #
            outpainted[0],
            smooth_edges[0],
            redo_background[0],
            final_refine[0],
        ]


@dataclass
class ModelInstance(BaseModelInstance[Params, "ProcessModels"]):
    model: Model

    def _generate_face_images(self) -> list[Image.Image]:
        latents = [
            self.model.models.pipe(
                image=img,
                generator=self.generator,
                num_inference_steps=self.params.steps,
                guidance_scale=self.params.guidance_scale,
                adapter_conditioning_scale=random.triangular(
                    *self.params.conditioning_strength
                ),
                adapter_conditioning_factor=self.params.conditioning_factor,
                output_type="latent",
                denoising_end=self.params.high_noise_frac + 0.1,
                **self.details_prompts.kwargs_for_xl(idx),
            ).images[0]
            for idx, img in enumerate(self.src_controls)
        ]

        return [
            self.model.models.base.refiner(
                image=latent,
                generator=self.generator,
                num_inference_steps=self.params.steps,
                denoising_start=self.params.high_noise_frac + 0.1,
                **self.details_prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, latent in enumerate(latents)
        ]

    @cached_property
    @collect
    def frames(self) -> Generator[Image.Image, None, None]:
        for idx, face in self.face_helper(self.faces, "faces").primary_face_bounds():
            bounds = Bounds.from_face(self.dims, face)
            self.model.logger.info("Face {} Bounds: {}", idx, bounds)
            target_percent = random.triangular(0.55, 0.60)

            curr_width, curr_height = bounds.size()
            target_width, target_height = [int(x * target_percent) for x in self.dims]
            d_w, d_h = target_width - curr_width, target_height - curr_height
            delta = max(d_w, d_h)
            padding = delta / max(self.dims)
            self.model.logger.info("Length Delta: {}, Padding: {}", delta, padding)

            face_img = np.asarray(self.faces[idx])[
                bounds.slice(self.params.mask_padding)
            ]
            embed = np.asarray(to_pil_image(face_img).resize(bounds.size(padding)))

            frame = np.zeros((*self.dims, 3), dtype=np.uint8)
            frame[bounds.slice(padding)] = embed
            yield to_pil_image(frame)

    def _base_background(self, idx: int) -> Image.Image:
        return self.models.base.base(
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
    def src_controls(self) -> Generator[Image.Image, None, None]:
        self.model.logger.info("Loading controls...")
        face_helper = self.face_helper(self.request.generation.images, "controls")
        for i, bounds in enumerate(self.request.generation.faces):
            face = face_helper._face_from_bounds(i, bounds)
            sharpened = unsharp_mask(np.asarray(face))
            yield self.models.detector(
                to_pil_image(sharpened),
                detect_resolution=self.params.detect_resolution,
                image_resolution=self.params.model.resolution,
            )

    @cached_property
    @collect
    def _outpaint_bases(
        self,
    ) -> Generator[
        tuple[Image.Image, tuple[Image.Image, Image.Image], Image.Image], None, None
    ]:
        bg_face_helper = self.face_helper(self.backgrounds, "backgrounds")
        bg_face_bounds = bg_face_helper.primary_face_bounds()

        frame_face_helper = self.face_helper(self.frames, "frames")
        frame_face_bounds = frame_face_helper.primary_face_bounds()

        for idx, frame_img in enumerate(self.frames):
            bg = self.backgrounds[idx]
            bg_face = bg_face_bounds[idx][1]
            bg_bounds = Bounds.from_face(self.dims, bg_face)

            frame = np.asarray(frame_img)
            frame_face = frame_face_bounds[idx][1]
            frame_bounds = Bounds.from_face(self.dims, frame_face)
            frame_mask, frame_mask_sm = self._get_masks(frame, frame_face)

            frame_mask = np.asarray(frame_mask.convert("RGB"))
            frame_mask_sm = np.asarray(frame_mask_sm.convert("RGB"))

            slice = (frame_mask == 0) & (frame != 0)
            masked_face = np.where(slice, frame, np.asarray(bg))

            self.model.logger.info(
                "Using bg_bounds: {}, fg_bounds: {}", bg_bounds, frame_bounds
            )

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

            self.model.logger.info(
                "{}",
                {
                    "idx": idx,
                    "bg": (Cx, Cy),
                    "fg": (Sx, Sy),
                    "start": (start_x, start_y),
                    "end": (end_x, end_y),
                    "bounds": frame_bounds.slice(),
                },
            )

            bg_image = np.array(bg)
            bg_image[start_y:end_y, start_x:end_x] = masked_face[frame_bounds.slice()]

            mask = np.full(bg_image.shape, 255, dtype=np.uint8)
            mask[start_y:end_y, start_x:end_x] = frame_mask[frame_bounds.slice()]

            mask_sm = np.full(bg_image.shape, 255, dtype=np.uint8)
            mask_sm[start_y:end_y, start_x:end_x] = frame_mask_sm[frame_bounds.slice()]

            control_data = np.asarray(
                self.request.generation.controls[idx]
                .resize(frame_bounds.size())
                .convert("RGB")
            )
            control = np.zeros(bg_image.shape, dtype=np.uint8)
            control[start_y:end_y, start_x:end_x] = control_data

            yield (
                to_pil_image(bg_image, mode="RGB"),
                (
                    to_pil_image(mask, mode="RGB").convert("L"),
                    to_pil_image(mask_sm, mode="RGB").convert("L"),
                ),
                to_pil_image(control, mode="RGB").convert("L"),
            )

    @cached_property
    @collect
    def controls(self) -> Generator[Image.Image, None, None]:
        yield from map(itemgetter(2), self._outpaint_bases)

    @cached_property
    @collect
    def _masks(self) -> Generator[tuple[Image.Image, Image.Image], None, None]:
        yield from map(itemgetter(1), self._outpaint_bases)

    @property
    def outpaint_bases(self) -> list[Image.Image]:
        return list(map(itemgetter(0), self._outpaint_bases))

    @collect
    def outpaint(self) -> Generator[Image.Image, None, None]:
        latents = [
            self.models.base.inpainter(
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
            self.models.base.refine_inpainter(
                image=latent,
                latents=latent,
                mask_image=self.mask_sms[idx],
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
            self.models.base.refiner(
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

    def _redo_background(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.base.inpainter(
                image=image,
                mask_image=self.masks[idx],
                generator=self.generator,
                strength=0.95,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.details_prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, image in enumerate(images)
        ]

    def _final_refine(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.base.refine_inpainter(
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
