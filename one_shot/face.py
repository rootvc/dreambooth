import itertools
from collections import Counter
from dataclasses import dataclass
from functools import cache as f_cache
from functools import cached_property, reduce
from operator import itemgetter
from pathlib import Path
from typing import Generator, Generic, Iterator, Optional, TypeVar, cast

import cv2
import numpy as np
import torch.amp
from loguru._logger import Logger
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from torchvision.transforms.functional import to_pil_image
from transformers import BlipForQuestionAnswering, BlipProcessor, SamModel, SamProcessor

import one_shot
from one_shot.params import Params, Settings
from one_shot.utils import (
    Face,
    NpBox,
    collect,
    detect_faces,
    dilate_mask,
    exclude,
    translate_mask,
)

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
        if face.mask:
            contours, _ = cv2.findContours(
                cv2.cvtColor(face.mask.arr, cv2.COLOR_RGB2GRAY),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contour = sorted(contours, key=cv2.contourArea)[-1]
            x, y, w, h = cv2.boundingRect(contour)
            one_shot.logger.info("Mask bounds: {}", (x, y, w, h))
        elif face.landmarks and face.landmarks.contour:
            x, y, w, h = cv2.boundingRect(
                np.asarray(list(map(list, face.landmarks.contour)), dtype=np.int32)
            )
            one_shot.logger.info("Landmark bounds: {}", (x, y, w, h))
        else:
            x = face.box.top_left[0]
            y = face.box.top_left[1]
            w = face.box.bottom_right[0] - face.box.top_left[0]
            h = face.box.bottom_right[1] - face.box.top_left[1]
            one_shot.logger.info("Face bounds: {}", (x, y, w, h))
        return cls(dims=shape, shape=shape, x=x, y=y, w=w, h=h)

    @property
    def center(self):
        return (self.origin_x + self.width // 2, self.origin_y + self.height // 2)

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

    def size(self, mask_padding: float = 0.0) -> tuple[float, float]:
        slice_y, slice_x = self.slice(mask_padding)
        return (slice_x.stop - slice_x.start, slice_y.stop - slice_y.start)

    def _slice(self, mask_padding: float) -> tuple[slice, slice]:
        buffer_y = int(self.dims[0] * mask_padding)
        buffer_x = int(self.dims[1] * mask_padding)

        start_y = max(0, self.origin_y - buffer_y)
        end_y = min(self.shape[0] - 1, self.origin_y + self.height + buffer_y)
        start_x = max(0, self.origin_x - buffer_x)
        end_x = min(self.shape[1] - 1, self.origin_x + self.width + buffer_x)

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
        logger: Logger | None = None,
        conservative: bool = True,
    ):
        self.params = params
        self.src_images = src_images or images
        self.dst_images = images

        self.rank = models.rank
        self.models = models

        self.logger = logger or one_shot.logger
        self.conservative = conservative

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

    def _run_sam(
        self, img: Image.Image, face: Face
    ) -> tuple[np.ndarray, list[np.float32]]:
        inputs = self.models.sam.processor(
            img,
            input_points=[[[img.width // 2, img.height // 2]]]
            if face.is_trivial
            else [[face.eyes.flat + (face.landmarks.flat if face.landmarks else [])]],
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
        mask_percentages = [
            np.count_nonzero(m) / np.prod(m.shape[:2], dtype=np.float32) for m in masks
        ]
        self.logger.info(
            "SAM scores: {}",
            [
                dict(zip(("score", "percent"), pair))
                for pair in zip(
                    outputs.iou_scores.cpu().numpy()[0][0], mask_percentages
                )
            ],
        )
        return masks, mask_percentages

    def _get_sam_mask(self, img: Image.Image, face: Face) -> np.ndarray | None:
        masks = [m for m, p in zip(*self._run_sam(img, face)) if p > 0.10]
        if len(masks) == 0:
            self.logger.warning("No SAM masks after filtering")
            return None
        self.logger.warning("Using SAM mask: {}", len(masks))
        mask = reduce(np.logical_or, masks) if self.conservative else masks[0]
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w, 1)

        if np.all(mask) or not np.any(mask):
            self.logger.warning("No SAM mask detected")
            return None

        zero = np.zeros(self.dims, dtype=np.uint8)
        one = np.full(self.dims, 255, dtype=np.uint8)
        mask_rbg = np.repeat(mask, 3, axis=2)
        mask = np.where(mask_rbg, one, zero)
        return mask

    def _get_landmark_mask(self, img: Image.Image, face: Face) -> np.ndarray | None:
        if not (face.landmarks and face.landmarks.contour):
            self.logger.warning("No landmarks")
            return None
        mask = np.zeros(self.dims, dtype=np.uint8)
        points = np.asarray(list(map(list, face.landmarks.contour)), dtype=np.int32)

        mean = np.mean(points, axis=0)
        angles = np.arctan2((points - mean)[:, 1], (points - mean)[:, 0])
        angles[angles < 0] = angles[angles < 0] + 2 * np.pi
        sorting_indices = np.argsort(angles)
        sorted_points = points[sorting_indices]
        cv2.drawContours(mask, [sorted_points], 0, (255, 255, 255), -1)

        if np.all(mask) or not np.any(mask):
            self.logger.warning("No mask detected")
        elif self.conservative:
            return dilate_mask(mask, iterations=25)
        else:
            mask = dilate_mask(mask, iterations=50)
            _, _, _, h = cv2.boundingRect(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY))
            return translate_mask(mask, (0, -h // 7))

    def _get_mask(self, img: Image.Image, face: Face) -> np.ndarray | None:
        fns = [self._get_landmark_mask, self._get_sam_mask]
        if self.conservative:
            fns.reverse()

        for fn in fns:
            if (mask := fn(img, face)) is not None:
                return mask
            else:
                self.logger.warning("No mask from {}", fn.__name__)

    @f_cache
    @collect
    def face_bounds(self) -> Generator[tuple[int, Face], None, None]:
        detections = [detect_faces(img) for img in self.src_images]
        for i, faces in enumerate(detections):
            img = self.src_images[i]
            for j, face in enumerate(faces):
                self.logger.info("Face: {}", face)
                if (mask := self._get_mask(img, face)) is not None:
                    face = face.copy(update={"mask": NpBox(arr=mask)})
                yield i, face

    @f_cache
    def _face_from_bounds(self, i: int, face: Face) -> Image.Image:
        img = np.array(self.dst_np_images[i])
        if face.mask:
            img = np.where(dilate_mask(face.mask.arr, iterations=25), img, 0)
        bounds = Bounds.from_face(self.dims, face)
        y, x = bounds.slice(self.params.mask_padding)
        image = img[y, x]
        return ImageOps.fit(to_pil_image(image), self.dims[:2])

    @collect
    def faces(self):
        for i, face in self.face_bounds():
            yield self._face_from_bounds(i, face)

    @collect
    def primary_face_bounds(self) -> Generator[tuple[int, Face], None, None]:
        for i, faces_with_index in itertools.groupby(self.face_bounds(), itemgetter(0)):
            faces: list[Face] = list(map(itemgetter(1), faces_with_index))
            selected = sorted(faces, reverse=True)[0]
            self.logger.warning("{}", {"face_options": faces, "selected": selected})
            yield i, selected

    @collect
    def primary_faces(self):
        for i, face in self.primary_face_bounds():
            yield self._face_from_bounds(i, face)

    @f_cache
    def demographics(self):
        self.logger.info("Analyzing demographics...")
        try:
            demos: dict[str, str] = {}
            for attr, vals in self.analyze().items():
                self.logger.info(f"{attr}: {vals}")
                demos[attr] = Counter(vals).most_common(1)[0][0]
            return demos
        except Exception as e:
            self.logger.exception(e)
            return {"ethnicity": "beautiful", "gender": "person"}

    @collect
    def eye_masks(self) -> Generator[tuple[Image.Image, str], None, None]:
        for i, faces in itertools.groupby(self.face_bounds(), itemgetter(0)):
            self.logger.info("Detecting eyes...")
            colors = []
            mask = np.zeros(self.dims, dtype=self.dst_np_images[i].dtype)
            for face in cast(Iterator[Face], map(itemgetter(1), faces)):
                for eye in face.eyes:
                    bounds = Bounds(
                        dims=self.dims, shape=self.dims, x=eye[0], y=eye[1], h=0, w=0
                    )
                    y, x = bounds._slice(0.05)
                    mask[y, x] = 255
                colors.append(
                    self._analyze([self._face_from_bounds(i, face)], "eye color")[0]
                )
            yield to_pil_image(mask), Counter(colors).most_common(1)[0][0]
