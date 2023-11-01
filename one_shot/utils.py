import gc
import hashlib
import itertools
import os
import shutil
import subprocess
import sys
from functools import wraps
from pathlib import Path
from typing import Callable, Generator, ParamSpec, TypeVar
from urllib.parse import urlencode

import cv2
import face_evolve.applications.align
import face_recognition.api as face_recognition
import numpy as np
import requests
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from pydantic import BaseModel
from pypdl import Downloader
from torch._inductor.autotune_process import tuning_process
from torch._inductor.codecache import AsyncCompile
from torchvision import transforms as TT
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import make_grid

from one_shot import logger
from one_shot.params import Params

sys.path.append(str(Path(face_evolve.applications.align.__file__).parent))
from face_evolve.applications.align.detector import (
    detect_faces as _detect_faces,  # noqa: E402
)

T = TypeVar("T")
P = ParamSpec("P")

TPoint = tuple[int, int]


class NpBox(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    arr: np.ndarray

    def __hash__(self) -> int:
        return int(hashlib.sha1(np.ascontiguousarray(self.arr)).hexdigest(), 16)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, NpBox):
            return False
        return np.array_equal(self.arr, value.arr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.arr.shape})"


class FrozenModel(BaseModel):
    class Config:
        frozen = True


class Box(FrozenModel):
    top_left: TPoint
    bottom_right: TPoint

    @property
    def flat(self):
        return [*self.top_left, *self.bottom_right]


class LR(FrozenModel):
    left: TPoint
    right: TPoint

    @property
    def flat(self):
        return [list(self.left), list(self.right)]

    def __iter__(self):
        yield self.left
        yield self.right


class Eyes(LR):
    pass


class Mouth(LR):
    pass


class Landmarks(FrozenModel):
    nose: TPoint
    mouth: Mouth
    contour: frozenset[TPoint] | None = None
    rest: frozenset[TPoint] | None = None

    @property
    def flat(self):
        return (
            self.mouth.flat
            + [list(self.nose)]
            + (list(map(list, self.rest)) if self.rest else [])
        )


class Face(FrozenModel):
    box: Box
    eyes: Eyes
    landmarks: Landmarks | None
    mask: NpBox | None = None

    @property
    def is_trivial(self) -> bool:
        return self.eyes.left == self.eyes.right == (0, 0)


def image_transforms(size: int) -> Callable[[Image.Image], Image.Image]:
    return TT.Compose(
        [
            TT.Resize(size, interpolation=TT.InterpolationMode.LANCZOS),
            # TT.ToTensor(),
            # TT.Normalize([0.5], [0.5]),
            # TT.ToPILImage(),
        ]
    )


def open_image(params: Params, path: Path) -> Image.Image:
    img = exif_transpose(Image.open(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return image_transforms(params.model.resolution)(img)


def collect(fn: Callable[P, Generator[T, None, None]]) -> Callable[P, list[T]]:
    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs):
        return list(fn(*args, **kwargs))

    return wrapped


def consolidate_cache(cache_dir: str, staging_dir: str):
    logger.warning(f"Consolidating cache from {staging_dir} to {cache_dir}")
    subprocess.run(
        [
            "rsync",
            "-ahSD",
            "--no-whole-file",
            "--no-compress",
            "--stats",
            "--inplace",
            f"{staging_dir}/",
            f"{cache_dir}/",
        ]
    )


def close_all_files(cache_dir: str):
    logger.warning(f"Closing all files with prefix {cache_dir}")

    AsyncCompile.pool().shutdown()
    AsyncCompile.process_pool().shutdown()
    tuning_process.terminate()
    gc.collect()
    logger.info("Closed all files")


def get_mtime(path: Path):
    try:
        return max(p.stat().st_mtime for p in path.rglob("*"))
    except ValueError:
        return path.stat().st_mtime


def civitai_path(model: str):
    return Path(os.environ["CACHE_DIR"]) / "civitai" / model / "model.safetensors"


def download_civitai_model(model: str):
    logger.info(f"Downloading CivitAI model {model}...")
    path = civitai_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    query = {"type": "Model", "format": "SafeTensor"}
    url = f"https://civitai.com/api/download/models/{model}?{urlencode(query)}"
    Downloader().start(url, str(path))
    if path.exists():
        return
    with requests.get(url, stream=True) as r:
        with open(path, "wb") as f:
            shutil.copyfileobj(r.raw, f)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def exclude(d: dict, keys: set[str]):
    return {k: v for k, v in d.items() if k not in keys}


def only(d: dict, keys: set[str]):
    return {k: v for k, v in d.items() if k in keys}


def images(p: Path):
    return list(itertools.chain(p.glob("*.jpg"), p.glob("*.png")))


def grid(images: list[Image.Image] | list[np.ndarray], w: int = 2) -> Image.Image:
    if any(np.asarray(image).shape[-1] == 4 for image in images):
        for image in images:
            if np.asarray(image).shape[-1] == 3:
                image.putalpha(255)
    tensors = torch.stack([to_tensor(img) for img in images])
    grid = make_grid(tensors, nrow=w, pad_value=255, padding=10)
    return to_pil_image(grid)


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    size, iterations = (5, 5), 15
    kernel = np.ones(size, np.uint8)
    return cv2.dilate(mask.copy(), kernel, iterations=iterations)


def erode_mask(mask: np.ndarray) -> np.ndarray:
    size, iterations = (3, 3), 35
    kernel = np.ones(size, np.uint8)
    return cv2.erode(mask.copy(), kernel, iterations=iterations)


@collect
def extract_faces(
    faces: list[tuple[np.ndarray, np.ndarray]], img: Image.Image
) -> Generator[Face, None, None]:
    for bbox, kps in faces:
        box = Box(
            top_left=(bbox[0].tolist(), bbox[1].tolist()),
            bottom_right=(bbox[2].tolist(), bbox[3].tolist()),
        )
        yield Face(
            box=box,
            eyes=Eyes(left=(kps[0], kps[0 + 5]), right=(kps[1], kps[1 + 5])),
            landmarks=_dlib_landmarks(
                np.asarray(img)[
                    box.top_left[1] : box.bottom_right[1],
                    box.top_left[0] : box.bottom_right[0],
                ]
            )
            or Landmarks(
                nose=(kps[2], kps[2 + 5]),
                mouth=Mouth(left=(kps[3], kps[3 + 5]), right=(kps[4], kps[4 + 5])),
            ),
        )


def _point_center(points: list[TPoint]) -> TPoint:
    return tuple(np.mean(points, axis=0).astype(int))


def __dlib_landmarks(image: np.ndarray) -> tuple[dict, Landmarks] | None:
    try:
        landmarks = face_recognition.face_landmarks(image)[0]
    except Exception:
        return None
    return landmarks, Landmarks(
        nose=_point_center(landmarks["nose_tip"]),
        mouth=Mouth(left=landmarks["top_lip"][0], right=landmarks["bottom_lip"][0]),
        contour=frozenset(
            filter(
                lambda p: all(x > 0 for x in p),
                itertools.chain.from_iterable(
                    only(landmarks, {"chin", "left_eyebrow", "right_eyebrow"}).values()
                ),
            )
        ),
        rest=frozenset(
            filter(
                lambda p: all(x > 0 for x in p),
                itertools.chain.from_iterable(landmarks.values()),
            )
        ),
    )


def _dlib_landmarks(image: np.ndarray) -> Landmarks | None:
    if landmarks := __dlib_landmarks(image):
        return landmarks[1]


@collect
def _dlib_detect_faces(img: Image.Image) -> Generator[Face, None, None]:
    for t, r, b, l in face_recognition.face_locations(np.asarray(img), model="cnn"):
        if (landmarks := __dlib_landmarks(np.asarray(img)[t:b, l:r])) is None:
            continue
        raw, parsed = landmarks
        yield Face(
            box=Box(top_left=(l, t), bottom_right=(r, b)),
            eyes=Eyes(
                left=_point_center(raw["left_eye"]),
                right=_point_center(raw["right_eye"]),
            ),
            landmarks=parsed,
        )


@collect
def _opencv_detect_faces(img: Image.Image) -> Generator[Face, None, None]:
    gray = np.array(img.convert("L"))
    smooth = cv2.GaussianBlur(gray, (125, 125), 0)
    image = cv2.divide(gray, smooth, scale=255)

    face_cascade = cv2.CascadeClassifier(
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        "/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml"
    )
    if len(
        faces := face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    ):
        for x, y, w, h in faces:
            eyes = iter(
                eye_cascade.detectMultiScale(
                    gray[y : y + h, x : x + w],
                    scaleFactor=1.3,
                    minNeighbors=10,
                    minSize=(30, 30),
                )
            )
            (xL, yL, wL, hL) = next(eyes, (0, 0, 0, 0))
            (xR, yR, wR, hR) = next(eyes, (0, 0, 0, 0))
            yield Face(
                box=Box(top_left=(x, y), bottom_right=(x + w, y + h)),
                eyes=Eyes(
                    left=(xL + wL // 2, yL + hL // 2),
                    right=(xR + wR // 2, yR + hR // 2),
                ),
                landmarks=None,
            )


@collect
def detect_faces(img: Image.Image) -> Generator[Face, None, None]:
    if faces := list(zip(*_detect_faces(img))):
        yield from extract_faces(faces, img)
    elif faces := _dlib_detect_faces(img):
        yield from faces
    elif faces := _opencv_detect_faces(img):
        yield from faces
    else:
        w, h = img.size
        yield Face(
            box=Box(top_left=(0, 0), bottom_right=(w, h)),
            eyes=Eyes(left=(0, 0), right=(0, 0)),
            landmarks=None,
        )
