from functools import wraps
from pathlib import Path
from typing import Callable, Generator, ParamSpec, TypeVar

import numpy as np
import torch
import torch._dynamo.config
import torch.backends.cuda
import torch.backends.cudnn
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms as TT

from dreambooth.train.helpers.face import FaceHelper
from dreambooth.train.shared import images as images
from one_shot.params import Params

T = TypeVar("T")
P = ParamSpec("P")


def set_torch_config():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch._dynamo.config.suppress_errors = True


def image_transforms(size: int) -> Callable[[Image.Image], torch.Tensor]:
    return TT.Compose(
        [
            TT.ToTensor(),
            TT.Resize(size, interpolation=TT.InterpolationMode.LANCZOS),
            TT.Normalize([0.5], [0.5]),
            TT.ToPILImage(),
        ]
    )


def open_image(params: Params, path: Path) -> np.ndarray:
    img = exif_transpose(Image.open(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = image_transforms(params.model.resolution)(img)
    return FaceHelper(params, img).mask()


def collect(fn: Callable[P, Generator[T, None, None]]) -> Callable[P, list[T]]:
    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs):
        return list(fn(*args, **kwargs))

    return wrapped
