import itertools
from collections import Counter
from dataclasses import dataclass
from functools import cache as f_cache
from functools import cached_property, reduce
from operator import itemgetter
from pathlib import Path
from typing import Generator, Generic, Iterator, Optional, TypeVar, cast

import numpy as np
import torch.amp
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from torchvision.transforms.functional import to_pil_image
from transformers import BlipForQuestionAnswering, BlipProcessor, SamModel, SamProcessor

from one_shot.params import Params, Settings
from one_shot.utils import Face, NpBox, collect, detect_faces, dilate_mask, exclude

M = TypeVar("M")
P = TypeVar("P")


settings = Settings()


class Bounds(BaseModel):
    dims: tuple[int, ...]
    shape: tuple[int, ...]

    origin_x: int = Field(int, alias="x")
    origin_y: int = Field(int, alias="y")
    width: int = Field(int, alias="w")
    height: int = Field(int, alias="h")

    @classmethod
    def from_face(cls, shape: tuple[int, int], face: Face):
        return cls(
            dims=shape,
            shape=shape,
            x=face.box.top_left[0],
            y=face.box.top_left[1],
            w=face.box.bottom_right[0] - face.box.top_left[0],
            h=face.box.bottom_right[1] - face.box.top_left[1],
        )

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

    def size(self, mask_padding: float = 0.0):
        slice_y, slice_x = self.slice(mask_padding)
        return (slice_x.stop - slice_x.start, slice_y.stop - slice_y.start)

    def _slice(self, mask_padding: float):
        buffer_y = int(self.dims[0] * mask_padding)
        buffer_x = int(self.dims[1] * mask_padding)

        start_y = max(0, self.origin_y - buffer_y)
        end_y = min(self.shape[0], self.origin_y + self.height + buffer_y)
        start_x = max(0, self.origin_x - buffer_x)
        end_x = min(self.shape[1], self.origin_x + self.width + buffer_x)

        return (slice(start_y, end_y), slice(start_x, end_x))

    def slice(self, mask_padding: float = 0.0):
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
class ModelAndProcessor(Generic[M, P]):
    model: M
    processor: P

    @classmethod
    def from_pretrained(
        cls, klasses: tuple[M, P], name: str, rank: int, half: bool = True, **kwargs
    ) -> "ModelAndProcessor[M, P]":
        Model, Processor = klasses
        model = Model.from_pretrained(name, **kwargs).to(
            rank, dtype=torch.float16 if half else None
        )
        processor = Processor.from_pretrained(name, **kwargs)
        return cls(model, processor)


@dataclass
class FaceHelperModels:
    rank: int
    blip: ModelAndProcessor[BlipForQuestionAnswering, BlipProcessor]
    sam: ModelAndProcessor[SamModel, SamProcessor]

    @classmethod
    def load(cls, params: Params, rank: int):
        kwargs = exclude(
            settings.loading_kwargs, {"use_safetensors", "variant", "torch_dtype"}
        )
        blip = ModelAndProcessor.from_pretrained(
            (BlipForQuestionAnswering, BlipProcessor),
            params.model.vqa,
            rank,
            torch_dtype=settings.loading_kwargs["torch_dtype"],
            **kwargs,
        )
        sam = ModelAndProcessor.from_pretrained(
            (SamModel, SamProcessor), params.model.sam, rank, half=False, **kwargs
        )
        return cls(rank=rank, blip=blip, sam=sam)


class FaceHelper:
    cache: Path = Path(settings.cache_dir) / "face"

    def __init__(
        self,
        params: Params,
        models: FaceHelperModels,
        images: list[Image.Image],
        src_images: Optional[list[Image.Image]] = None,
    ):
        self.params = params
        self.src_images = src_images or images
        self.dst_images = images

        self.rank = models.rank
        self.models = models

    def with_images(self, images: list[Image.Image]):
        return self.__class__(
            self.params,
            FaceHelperModels(self.rank, self.models.blip, self.models.sam),
            images,
            self.src_images,
        )

    @cached_property
    def dst_np_images(self):
        return list(map(np.asarray, self.dst_images))

    @collect
    def _analyze(
        self, images: list[Image.Image], attr: str
    ) -> Generator[str, None, None]:
        inputs = self.models.blip.processor(
            images,
            [
                f"What is the {attr} of the person? Answer with only one word/phrase and nothing else."
            ]
            * len(images),
            return_tensors="pt",
        ).to(self.rank, self.params.dtype)
        outputs = self.models.blip.model.generate(**inputs, max_length=50)
        yield from self.models.blip.processor.batch_decode(
            outputs, skip_special_tokens=True
        )

    @f_cache
    def analyze(self):
        results: dict[str, list[str]] = {}
        for attr in ("ethnicity", "gender"):
            results[attr] = self._analyze(self.primary_faces(), attr)
        return results

    @property
    def dims(self):
        return np.asarray(self.src_images[0]).shape

    @f_cache
    @collect
    def face_bounds(self) -> Generator[tuple[int, Face], None, None]:
        detections = [detect_faces(img) for img in self.src_images]
        for i, faces in enumerate(detections):
            img = self.src_images[i]
            for j, face in enumerate(faces):
                inputs = self.models.sam.processor(
                    img,
                    input_points=[[[img.width // 2, img.height // 2]]]
                    if face.is_trivial
                    else [
                        [
                            face.eyes.flat
                            + (face.landmarks.flat if face.landmarks else [])
                        ]
                    ],
                    input_boxes=None if face.is_trivial else [[face.box.flat]],
                    return_tensors="pt",
                ).to(self.rank)
                outputs = self.models.sam.model(**inputs)
                masks = self.models.sam.processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )
                masks = masks[0].cpu().numpy().squeeze()
                masks = [
                    m for m in masks if np.count_nonzero(m) > (0.10 * np.prod(m.shape))
                ]
                if len(masks) == 0:
                    continue
                mask_images = []
                for mask in (masks[0], reduce(np.logical_or, masks)):
                    h, w = mask.shape[-2:]
                    mask = mask.reshape(h, w, 1)

                    if np.all(mask) or not np.any(mask):
                        logger.warning("No mask detected")
                        continue

                    zero = np.zeros(self.dims, dtype=np.uint8)
                    one = np.full(self.dims, 255, dtype=np.uint8)
                    mask_rbg = np.repeat(mask, 3, axis=2)
                    mask_image = np.where(mask_rbg, one, zero)
                    mask_images.append(mask_image)

                faces[j] = face.copy(
                    update={
                        "aggressive_mask": NpBox(arr=dilate_mask(mask_images[0])),
                        "mask": NpBox(arr=dilate_mask(mask_images[1])),
                    }
                )
        for i, faces in enumerate(detections):
            for face in faces:
                yield i, face

    @f_cache
    def _face_from_bounds(self, i: int, face: Face) -> Image.Image:
        img = np.array(self.dst_np_images[i])
        if face.mask:
            img = np.where(face.mask.arr, img, 0)
        bounds = Bounds.from_face(self.dims, face)
        y, x = bounds.slice(self.params.mask_padding)
        image = self.dst_np_images[i][y, x]
        return to_pil_image(image).resize(self.dims[:2])

    @collect
    def faces(self):
        for i, face in self.face_bounds():
            yield self._face_from_bounds(i, face)

    @collect
    def primary_face_bounds(self):
        for i, faces in itertools.groupby(self.face_bounds(), itemgetter(0)):
            yield i, next(faces)[1]

    @collect
    def primary_faces(self):
        for i, face in self.primary_face_bounds():
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
            return {"ethnicity": "beautiful", "gender": "person"}

    @collect
    def eye_masks(self) -> Generator[tuple[Image.Image, str], None, None]:
        for i, faces in itertools.groupby(self.face_bounds(), itemgetter(0)):
            logger.info("Detecting eyes...")
            colors = []
            mask = np.zeros(self.dims, dtype=self.dst_np_images[i].dtype)
            for face in cast(Iterator[Face], map(itemgetter(1), faces)):
                logger.debug("Face: {}", face)
                for eye in face.eyes:
                    bounds = Bounds(
                        dims=self.dims, shape=self.dims, x=eye[0], y=eye[1], h=0, w=0
                    )
                    y, x = bounds._slice(self.params.mask_padding)
                    mask[y, x] = 255
                colors.append(
                    self._analyze([self._face_from_bounds(i, face)], "eye color")[0]
                )
            yield to_pil_image(mask), Counter(colors).most_common(1)[0][0]
