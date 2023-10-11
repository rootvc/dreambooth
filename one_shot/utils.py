import gc
import itertools
import os
import subprocess
from functools import wraps
from pathlib import Path
from typing import Callable, Generator, ParamSpec, TypeVar
from urllib.parse import urlencode

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


class Eyes(FrozenModel):
    left: tuple[int, int]
    right: tuple[int, int]

    def __iter__(self):
        yield self.left
        yield self.right


class Face(FrozenModel):
    box: Box
    eyes: Eyes


def extract_face(face) -> Face:
    return Face(
        box=Box(
            top_left=(face.bbox[0].tolist(), face.bbox[1].tolist()),
            bottom_right=(face.bbox[2].tolist(), face.bbox[3].tolist()),
        ),
        eyes=Eyes(left=face.kps[0].tolist(), right=face.kps[1].tolist()),
    )


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


def exclude(d: dict, keys: set[str]):
    return {k: v for k, v in d.items() if k not in keys}


def images(p: Path):
    return list(itertools.chain(p.glob("*.jpg"), p.glob("*.png")))


def grid(images: list[Image.Image] | list[np.ndarray], w: int = 2) -> Image.Image:
    tensors = torch.stack([to_tensor(img) for img in images])
    grid = make_grid(tensors, nrow=w, pad_value=255, padding=10)
    return to_pil_image(grid)
