import gc
import subprocess
from functools import wraps
from pathlib import Path
from typing import Callable, Generator, ParamSpec, TypeVar

from deepface import DeepFace
from dreambooth_old.train.shared import images as images
from loguru import logger
from PIL import Image
from PIL.ImageOps import exif_transpose
from retinaface import RetinaFace
from torch._inductor.autotune_process import tuning_process
from torch._inductor.codecache import AsyncCompile
from torchvision import transforms as TT

from one_shot.params import Params

T = TypeVar("T")
P = ParamSpec("P")


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

    if hasattr(DeepFace, "model_obj"):
        delattr(DeepFace, "model_obj")
    if hasattr(RetinaFace, "model"):
        delattr(RetinaFace, "model")

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
