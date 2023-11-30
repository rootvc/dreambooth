from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
from torchvision.transforms.functional import (
    to_pil_image,
)

from one_shot.face import Bounds, FaceHelper
from one_shot.params.base import Params
from one_shot.params.prompts import PromptStrings
from one_shot.params.settings import Settings
from one_shot.prompt import Prompts
from one_shot.utils import (
    Face,
    dilate_mask,
    erode_mask,
)

if TYPE_CHECKING:
    from one_shot.dreambooth.process.models.base import ProcessModelImpl
    from one_shot.dreambooth.process.process import ProcessRequest
    from one_shot.logging import PrettyLogger


P = TypeVar("P", bound="Params")
PM = TypeVar("PM", bound="ProcessModelImpl")


@dataclass
class Model(ABC, Generic[P, PM]):
    rank: int
    logger: "PrettyLogger"
    params: P
    models: PM
    settings: Settings = Settings()

    @torch.inference_mode()
    def run(self, request: "ProcessRequest") -> list[Image.Image]:
        return ModelInstance(self, request).generate()

    @abstractmethod
    def _tune(self, request: "ProcessRequest") -> list[Image.Image]:
        ...

    @torch.inference_mode()
    def tune(self, request: "ProcessRequest") -> list[Image.Image]:
        return self._tune(request)


@dataclass
class ModelInstance(ABC, Generic[P, PM]):
    model: Model[P, PM]
    request: "ProcessRequest"

    def face_helper(self, images: list[Image.Image], tag: str = "", **kwargs):
        return FaceHelper(
            self.params,
            self.models.base.face,
            images,
            logger=self.model.logger.bind(tag=tag),
            **kwargs,
        )

    @property
    def rank(self) -> int:
        return self.model.rank

    @property
    def models(self) -> PM:
        return self.model.models

    @property
    def params(self) -> P:
        return self.model.params

    def _prompts(self, strings: PromptStrings, **kwargs) -> Prompts:
        return Prompts(
            self.models.base.compels,
            self.rank,
            self.params.dtype,
            [
                strings.positive(prompt=p, **self.request.demographics.dict(), **kwargs)
                for p in self.request.generation.prompts
            ],
            [
                strings.negative(prompt=p, **self.request.demographics.dict(), **kwargs)
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

    @abstractmethod
    def _generate_face_images(self) -> list[Image.Image]:
        ...

    def _touch_up_eyes(self, images: list[Image.Image]) -> list[Image.Image]:
        face_helper = self.face_helper(images, "eyes")
        masks, colors = map(list, zip(*face_helper.eye_masks()))
        self.model.logger.info("Colors: {}", colors)
        return [
            self.models.base.inpainter(
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

    def _get_masks(
        self, frame: np.ndarray, face: Face
    ) -> tuple[Image.Image, Image.Image]:
        if face.mask:
            self.model.logger.warning("Using face mask")
            mask = ~face.mask.arr
            mask_sm = ~erode_mask(face.mask.arr)
        elif not face.is_trivial:
            self.model.logger.warning("Using bounds mask")
            mask = np.full(frame.shape, 255, dtype=np.uint8)
            mask[
                Bounds.from_face(frame.shape[:2], face).slice(self.params.mask_padding)
            ] = 0
            mask_sm = np.full(frame.shape, 255, dtype=np.uint8)
            mask_sm[Bounds.from_face(frame.shape[:2], face).slice()] = 0
            mask_sm = ~erode_mask(~mask_sm)
        else:
            self.model.logger.warning("Using default mask")
            mask = np.full(frame.shape, 255, dtype=np.uint8)
            mask[frame != 0] = 0
            mask_sm = ~erode_mask(~mask)

        return tuple(to_pil_image(m, mode="RGB").convert("L") for m in (mask, mask_sm))

    @property
    @abstractmethod
    def _masks(self) -> list[tuple[Image.Image, Image.Image]]:
        ...

    @property
    def masks(self) -> list[Image.Image]:
        return list(map(itemgetter(0), self._masks))

    @property
    def mask_sms(self) -> list[Image.Image]:
        return list(map(itemgetter(1), self._masks))

    @cached_property
    def edge_masks(self) -> list[Image.Image]:
        return [
            to_pil_image(
                (
                    dilate_mask(~np.asarray(self.masks[idx]))
                    - ~np.asarray(self.mask_sms[idx])
                )
            )
            for idx in range(len(self._masks))
        ]

    @abstractmethod
    def generate(self) -> list[Image.Image]:
        ...
