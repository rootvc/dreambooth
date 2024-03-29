import gc
import hashlib
import itertools
import os
import random
import shutil
import subprocess
import sys
from collections import UserList
from functools import (
    cached_property,
    lru_cache,
    partial,
    total_ordering,
    update_wrapper,
    wraps,
)
from pathlib import Path
from typing import Callable, Generator, Generic, Optional, ParamSpec, TypeVar
from urllib.parse import urlencode

import cv2
import face_evolve.applications.align
import face_recognition.api as face_recognition
import numpy as np
import requests
import rich.repr
import torch
from huggingface_hub import hf_hub_download
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
from face_evolve.applications.align.detector import (  # noqa: E402
    detect_faces as _detect_faces,
)

T = TypeVar("T")
P = ParamSpec("P")

TPoint = tuple[int, int]

MASK_SIZE = Params().model.resolution ** 2


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

    def __rich_repr__(self) -> rich.repr.Result:
        yield "shape", self.arr.shape
        yield "mask", round(
            np.count_nonzero(self.arr) / np.prod(self.arr.shape), 2
        ) * 100


class FrozenModel(BaseModel):
    class Config:
        frozen = True


class Box(FrozenModel):
    top_left: TPoint
    bottom_right: TPoint

    @property
    def flat(self):
        return [*self.top_left, *self.bottom_right]

    def size(self):
        return (self.bottom_right[0] - self.top_left[0]) * (
            self.bottom_right[1] - self.top_left[1]
        )


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


@total_ordering
class Face(FrozenModel):
    box: Box
    eyes: Eyes
    landmarks: Landmarks | None
    mask: NpBox | None = None

    @property
    def is_trivial(self) -> bool:
        return self.eyes.left == self.eyes.right == (0, 0)

    def __rich_repr__(self) -> rich.repr.Result:
        yield "box", self.box
        yield "eyes", self.eyes
        yield "mask", self.mask
        yield "sort", self.__sort_features__()

    def __sort_features__(self) -> tuple:
        frac = self.box.size() / MASK_SIZE
        return (
            0 if self.is_trivial else 1,
            0 if self.mask is None else 1,
            1 if self.landmarks and self.landmarks.contour else 0,
            1 if (1 / 3 <= frac <= 1 / 2) else 0,
            0 if self.landmarks is None else 1,
            frac,
        )

    def __lt__(self, other):
        if not isinstance(other, Face):
            return NotImplemented
        return self.__sort_features__() < other.__sort_features__()


def method_cache(
    method: Optional[Callable[..., T]],
    *,
    maxsize: Optional[int] = 128,
    typed: bool = False,
) -> Callable[..., T]:
    def decorator(wrapped: Callable[..., T]) -> Callable[..., T]:
        def wrapper(self: object) -> Callable[..., T]:
            return lru_cache(maxsize=maxsize, typed=typed)(
                update_wrapper(partial(wrapped, self), wrapped)
            )

        return cached_property(wrapper)  # type: ignore

    return decorator if method is None else decorator(method)


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


def mask_bounding_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask.copy(), cv2.COLOR_RGB2GRAY)
    x, y, w, h = cv2.boundingRect(mask)
    return (x, y, w, h)


def mask_dilate_kernel(mask: np.ndarray) -> np.ndarray:
    _, _, w, h = mask_bounding_rect(mask)
    if w > h:
        ratio = min(w / h, 3)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(ratio * 3), 3))
    else:
        ratio = min(h / w, 3)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, round(ratio * 3)))


def dilate_mask(
    mask: np.ndarray, iterations: int = 15, kernel: np.ndarray | None = None
) -> np.ndarray:
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)
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
                np.asarray(img),
                (
                    box.top_left[1],
                    box.bottom_right[0],
                    box.bottom_right[1],
                    box.top_left[0],
                ),
            )
            or Landmarks(
                nose=(kps[2], kps[2 + 5]),
                mouth=Mouth(left=(kps[3], kps[3 + 5]), right=(kps[4], kps[4 + 5])),
            ),
        )


def _point_center(points: list[TPoint]) -> TPoint:
    return tuple(np.mean(points, axis=0).astype(int))


def __dlib_landmarks(
    image: np.ndarray, loc: tuple[int, int, int, int]
) -> tuple[dict, Landmarks] | None:
    try:
        landmarks = face_recognition.face_landmarks(image, [loc])[0]
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


def _dlib_landmarks(
    image: np.ndarray, loc: tuple[int, int, int, int]
) -> Landmarks | None:
    if landmarks := __dlib_landmarks(image, loc):
        return landmarks[1]


@collect
def _dlib_detect_faces(img: Image.Image) -> Generator[Face, None, None]:
    for t, r, b, l in face_recognition.face_locations(np.asarray(img), model="cnn"):
        if (landmarks := __dlib_landmarks(np.asarray(img), (t, r, b, l))) is None:
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
def _detect_dlib_faces(img: Image.Image) -> Generator[Face, None, None]:
    try:
        if faces := list(zip(*_detect_faces(img))):
            yield from extract_faces(faces, img)
    except Exception:
        return


@collect
def detect_faces(img: Image.Image) -> Generator[Face, None, None]:
    if faces := _detect_dlib_faces(img):
        yield from faces
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


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def load_hf_file(repo: str, path: str | Path, **kwargs):
    path = Path(path)
    return hf_hub_download(repo, path.name, subfolder=str(path.parent), **kwargs)


def draw_masks(img: Image.Image, faces: list[Face]):
    out = np.array(img)
    for face in faces:
        if not face.mask:
            continue
        color = np.array(
            [0, random.randint(0, 255), 0],
            dtype="uint8",
        )
        img2 = np.where(face.mask.arr, color, out)
        out = cv2.addWeighted(out, 0.6, img2, 0.4, 0)
    return Image.fromarray(out)


def translate_mask(mask: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    mat = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    return cv2.warpAffine(mask, mat, mask.shape[:2])


X = TypeVar("X")


class SelectableList(UserList, Generic[X]):
    def __init__(self, data: list[X]):
        super().__init__(data)

    def select(self, klass: type[X]) -> X:
        return next(x for x in self.data if isinstance(x, klass))


class Demographics(BaseModel):
    ethnicity: str
    gender: str
    facing_forwards: bool
    dark_skinned: bool

    def use_ip_adapter(self, faces: list[Face]) -> bool:
        return (
            self.is_white_person()
            or not self.facing_forwards
            or any(f.landmarks is None for f in faces)
        )

    def is_white_person(self):
        return not self.dark_skinned or self.ethnicity.lower().strip() in {
            "white",
            "caucasian",
        }
