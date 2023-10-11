import itertools
from collections import Counter
from dataclasses import dataclass
from functools import cache as f_cache
from functools import cached_property
from operator import itemgetter
from pathlib import Path
from typing import Generator, Iterator, cast

import numpy as np
import snoop
import torch
from insightface.app import FaceAnalysis
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from torchvision.transforms.functional import to_pil_image
from transformers import ViltForQuestionAnswering, ViltProcessor

from one_shot.params import Params, Settings
from one_shot.utils import Face, collect, extract_face, grid

settings = Settings()


class Bounds(BaseModel):
    dims: tuple[int, ...]
    shape: tuple[int, ...]

    origin_x: int = Field(int, alias="x")
    origin_y: int = Field(int, alias="y")
    width: int = Field(int, alias="w")
    height: int = Field(int, alias="h")

    @property
    def mid_y(self):
        return self.shape[0] // 2

    @property
    def mid_x(self):
        return self.shape[1] // 2

    @property
    def quadrant(self):
        if self.origin_y < self.mid_y:
            if self.origin_x < self.mid_x:
                return 0
            else:
                return 1
        else:
            if self.origin_x < self.mid_x:
                return 2
            else:
                return 3

    def _slice(self, mask_padding: float):
        buffer_y = int(self.dims[0] * mask_padding)
        buffer_x = int(self.dims[1] * mask_padding)

        start_y = max(0, self.origin_y - buffer_y)
        end_y = min(self.shape[0], self.origin_y + self.height + buffer_y)
        start_x = max(0, self.origin_x - buffer_x)
        end_x = min(self.shape[1], self.origin_x + self.width + buffer_x)

        return (slice(start_y, end_y), slice(start_x, end_x))

    def slice(self, mask_padding: float):
        original_slices = self._slice(mask_padding)
        quadrant = self.quadrant

        if quadrant == 0:
            return original_slices
        elif quadrant == 1:
            return (original_slices[0], slice(0, original_slices[1].stop - self.mid_x))
        elif quadrant == 2:
            return (slice(0, original_slices[0].stop - self.mid_y), original_slices[1])
        elif quadrant == 3:
            return (
                slice(0, original_slices[0].stop - self.mid_y),
                slice(0, original_slices[1].stop - self.mid_x),
            )
        else:
            raise ValueError(f"Invalid quadrant: {quadrant}")


@dataclass
class FaceHelperModels:
    MODEL = "dandelin/vilt-b32-finetuned-vqa"

    processor: ViltProcessor
    model: ViltForQuestionAnswering
    detector: FaceAnalysis

    @classmethod
    def load(cls, rank: int):
        kwargs = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        processor = ViltProcessor.from_pretrained(cls.MODEL, **kwargs)
        model = ViltForQuestionAnswering.from_pretrained(cls.MODEL, **kwargs)

        cache_path = Path(settings.cache_dir) / "insightface"
        detector = FaceAnalysis(
            root=str(cache_path / "models"),
            providers=[
                (
                    "OpenVINOExecutionProvider",
                    {"device_type": "CPU_FP16", "cache_dir": cache_path / "vino_cache"},
                ),
                "CPUExecutionProvider",
            ],
            allowed_modules=["detection", "landmark_2d_106", "landmark_3d_68"],
        )
        detector.prepare(rank)

        return cls(processor, model, detector)


class FaceHelper:
    MODEL = "dandelin/vilt-b32-finetuned-vqa"

    cache: Path = Path(settings.cache_dir) / "face"

    def __init__(
        self, params: Params, models: FaceHelperModels, images: list[Image.Image]
    ):
        self.params = params
        self.images = images
        self.processor, self.model, self.detector = [
            models.processor,
            models.model,
            models.detector,
        ]

    @cached_property
    def image(self):
        return np.asarray(grid(self.images))

    @cached_property
    def np_images(self):
        return list(map(np.asarray, self.images))

    @collect
    def _analyze(
        self, images: list[Image.Image], attr: str
    ) -> Generator[str, None, None]:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            encoding = self.processor(
                images,
                [f"What is the {attr} of the person?"] * len(images),
                return_tensors="pt",
            )
            outputs = self.model(**encoding)
            for logits in outputs.logits:
                idx = logits.argmax(-1).item()
                yield self.model.config.id2label[idx]

    @f_cache
    def analyze(self):
        results: dict[str, list[str]] = {}
        for attr in ("race", "gender"):
            results[attr] = self._analyze(self.images, attr)
        return results

    @property
    def dims(self):
        return np.asarray(self.images[0]).shape

    @f_cache
    @collect
    def face_bounds(self) -> Generator[tuple[int, Face], None, None]:
        detections = [
            map(extract_face, self.detector.get(img)) for img in self.np_images
        ]
        for i, faces in enumerate(detections):
            for face in faces:
                yield i, face

    @f_cache
    def _face_from_bounds(self, i: int, face: Face) -> Image.Image:
        bounds = Bounds(
            dims=self.dims,
            shape=self.dims,
            x=face.box.top_left[0],
            y=face.box.top_left[1],
            w=face.box.bottom_right[0] - face.box.top_left[0],
            h=face.box.bottom_right[1] - face.box.top_left[1],
        )
        y, x = bounds.slice(self.params.mask_padding)
        masked = np.zeros(self.dims, dtype=self.np_images[i].dtype)
        masked[y, x] = self.np_images[i][y, x]
        return to_pil_image(masked)

    @collect
    def faces(self):
        for i, face in self.face_bounds():
            yield self._face_from_bounds(i, face)

    @f_cache
    def demographics(self):
        logger.info("Analyzing demographics...")
        try:
            demos: dict[str, str] = {}
            for attr, vals in self.analyze().items():
                logger.info(f"{attr}: {vals}")
                demos[attr] = Counter(vals).most_common(1)[0][0]
            return demos
        except Exception as e:
            logger.exception(e)
            return {"race": "beautiful", "gender": "person"}

    @collect
    def eye_masks(self) -> Generator[tuple[Image.Image, str], None, None]:
        for i, faces in itertools.groupby(self.face_bounds(), itemgetter(0)):
            logger.info("Detecting eyes...")
            colors = []
            mask = np.zeros(self.dims, dtype=self.np_images[i].dtype)
            for face in cast(Iterator[Face], map(itemgetter(1), faces)):
                logger.debug("Face: {}", face)
                for eye in face.eyes:
                    bounds = Bounds(
                        dims=self.dims, shape=self.dims, x=eye[0], y=eye[1], h=0, w=0
                    )
                    y, x = bounds._slice(self.params.mask_padding * 1.25)
                    mask[y, x] = 255
                colors.append(
                    snoop.pp.deep(
                        lambda: self._analyze(
                            [self._face_from_bounds(i, face)], "eye color"
                        )[0]
                    )
                )
            yield to_pil_image(mask), Counter(colors).most_common(1)[0][0]
