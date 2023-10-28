import gc
import hashlib
import itertools
import os
import subprocess
from functools import wraps
from pathlib import Path
from typing import Callable, Generator, ParamSpec, TypeVar
from urllib.parse import urlencode

import cv2
import numpy as np
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

T = TypeVar("T")
P = ParamSpec("P")


class FrozenModel(BaseModel):
    class Config:
        frozen = True


class Box(FrozenModel):
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]

    @property
    def flat(self):
        return [*self.top_left, *self.bottom_right]


class Eyes(FrozenModel):
    left: tuple[int, int]
    right: tuple[int, int]

    @property
    def flat(self):
        return [list(self.left), list(self.right)]

    def __iter__(self):
        yield self.left
        yield self.right


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


class Face(FrozenModel):
    box: Box
    eyes: Eyes
    mask: NpBox | None = None
    raw_mask: NpBox | None = None

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
    Downloader().start(
        f"https://civitai.com/api/download/models/{model}?{urlencode(query)}", str(path)
    )


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def exclude(d: dict, keys: set[str]):
    return {k: v for k, v in d.items() if k not in keys}


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


def dilate_mask(mask: np.ndarray, strong: bool = True) -> np.ndarray:
    if strong:
        size, iterations = (5, 5), 20
    else:
        size, iterations = (4, 4), 20
    kernel = np.ones(size, np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


@collect
def extract_faces(
    points: tuple[list[np.ndarray], list[np.ndarray]], img: Image.Image
) -> Generator[Face, None, None]:
    faces = list(zip(*points))
    if not faces:
        w, h = img.size
        yield Face(
            box=Box(top_left=(0, 0), bottom_right=(w, h)),
            eyes=Eyes(left=(0, 0), right=(0, 0)),
        )
        return
    for bbox, kps in faces:
        yield Face(
            box=Box(
                top_left=(bbox[0].tolist(), bbox[1].tolist()),
                bottom_right=(bbox[2].tolist(), bbox[3].tolist()),
            ),
            eyes=Eyes(left=(kps[0], kps[0 + 5]), right=(kps[1], kps[1 + 5])),
        )
