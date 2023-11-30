import random
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Generator

import numpy as np
from PIL import Image

from one_shot.dreambooth.request.model.base import Model as BaseModel
from one_shot.dreambooth.request.model.base import ModelInstance as BaseModelInstance
from one_shot.params.ip_adapter import Params
from one_shot.utils import (
    collect,
)

if TYPE_CHECKING:
    from one_shot.dreambooth.process.models.ip_adapter import ProcessModels
    from one_shot.dreambooth.process.process import ProcessRequest


@dataclass
class Model(BaseModel):
    models: "ProcessModels"
    params: Params

    def _tune(self, request: "ProcessRequest") -> list[Image.Image]:
        instance = ModelInstance(self, request)
        faces = instance.faces
        redo_background = instance._redo_background(faces)
        final_refine = instance._final_refine(redo_background)
        return [
            instance.request.generation.images[0],
            faces[0],
            redo_background[0],
            final_refine[0],
        ]


@dataclass
class ModelInstance(BaseModelInstance[Params, "ProcessModels"]):
    model: Model

    def _generate_face_images(self) -> list[Image.Image]:
        return [
            self.model.models.ip_adapter.generate(
                image=img,
                generator=self.generator,
                num_inference_steps=self.params.steps,
                guidance_scale=self.params.guidance_scale,
                control_guidance_start=1.0 - self.params.conditioning_factor,
                scale=random.triangular(*self.params.conditioning_strength),
                **self.details_prompts.kwargs_for_base(idx),
            )[0].resize(self.dims)
            for idx, img in enumerate(self.request.generation.images)
        ]

    def _redo_background(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.base.inpainter(
                image=image,
                mask_image=self.masks[idx],
                generator=self.generator,
                strength=0.99,
                guidance_scale=self.params.refine_guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.details_prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, image in enumerate(images)
        ]

    def _final_refine(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.base.refine_inpainter(
                image=img,
                mask_image=self.masks[idx],
                generator=self.generator,
                strength=self.params.refiner_strength,
                guidance_scale=self.params.refine_guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.merge_prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]

    @cached_property
    @collect
    def _masks(self) -> Generator[tuple[Image.Image, Image.Image], None, None]:
        face_helper = self.face_helper(self.faces, "faces", conservative=False)
        bounds = face_helper.primary_face_bounds()
        for idx, face in enumerate(self.faces):
            yield self._get_masks(np.asarray(face), bounds[idx][1])

    def generate(self) -> list[Image.Image]:
        redo_background = self._redo_background(self.faces)
        return self._final_refine(redo_background)
