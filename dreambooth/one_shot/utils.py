import gc
import os
import subprocess
from functools import wraps
from pathlib import Path
from typing import Callable, Generator, ParamSpec, TypeVar

import psutil
from deepface import DeepFace
from dreambooth_old.train.helpers.face import FaceHelper
from dreambooth_old.train.shared import images as images
from loguru import logger
from PIL import Image
from PIL.ImageOps import exif_transpose
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


def close_all_files(prefix: str):
    logger.warning(f"Closing all files with prefix {prefix}")
    subprocess.run(["ls", "-la", prefix])
    subprocess.run(["ps", "afx"])

    if hasattr(DeepFace, "model_obj"):
        delattr(DeepFace, "model_obj")
    if hasattr(FaceHelper, "_face_detector"):
        delattr(FaceHelper, "_face_detector")
    if hasattr(DeepFace, "model_obj"):
        delattr(DeepFace, "model_obj")
    gc.collect()

    def close_all_files(p):
        for f in p.open_files():
            if not f.path.startswith(prefix):
                continue
            logger.warning(f"Closing {f.path}")
            try:
                os.close(f.fd)
            except OSError:
                pass

    process = psutil.Process()
    for child in process.children(recursive=True):
        logger.warning(child)
        close_all_files(child)
    close_all_files(process)

    subprocess.run(["lsof"])

    # os.closerange(3, os.sysconf("SC_OPEN_MAX"))


def get_mtime(path: Path):
    try:
        return max(p.stat().st_mtime for p in path.rglob("*"))
    except ValueError:
        return path.stat().st_mtime
