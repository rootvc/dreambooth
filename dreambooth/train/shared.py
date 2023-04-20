import functools
import importlib
import itertools
import logging
from contextlib import contextmanager
from ctypes import c_int
from datetime import datetime
from hashlib import sha1
from inspect import isfunction
from pathlib import Path
from typing import Callable, Hashable, Optional, TypeVar

import cv2
import numpy as np
import orjson as json
import torch
import torch.distributed
from PIL import Image
from torch._dynamo.eval_frame import OptimizedModule
from torchvision import transforms as TT
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import PushToHubMixin

T = TypeVar("T")
M = TypeVar("M", bound=torch.nn.Module)


def partition(
    d: dict[str, T], pred: Callable[[tuple[str, T]], bool]
) -> tuple[dict[str, T], dict[str, T]]:
    t1, t2 = itertools.tee(d.items())
    return tuple(map(dict, [itertools.filterfalse(pred, t1), filter(pred, t2)]))


def main_process_only(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if torch.distributed.is_initialized() and not self.accelerator.is_main_process:
            return
        return f(self, *args, **kwargs)

    return wrapper


@contextmanager
def patch_allowed_pipeline_classes():
    from diffusers import pipelines

    LIBS = ["diffusers", "transformers"]

    for lib in LIBS:
        setattr(pipelines, lib, importlib.import_module(lib))

    logging.disable()
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)

    for lib in LIBS:
        delattr(pipelines, lib)


def local_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main():
    return local_rank() == 0


multiprocessing = torch.multiprocessing.get_context("forkserver")
val = multiprocessing.Value(c_int)


def init_process(world_size: int):
    with val.get_lock():
        val.value += 1
    torch.distributed.init_process_group(rank=val.value, world_size=world_size)


def convert_image(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def compile_model(
    model: M,
    do: bool = True,
    ignore: set[str] = set(),
    backend: Optional[str] = "inductor",
    **kwargs,
) -> M:
    BROKEN_COMPILE_CLASSES = {"PreTrainedTokenizer", "CLIPTokenizer"}

    if isinstance(model, OptimizedModule):
        raise RuntimeError("Model is already compiled")
    elif isfunction(model):
        raise RuntimeError("Model is a function")

    if not backend:
        return model
    if model.__class__.__name__ in BROKEN_COMPILE_CLASSES:
        return model
    elif model.__class__.__name__ in ignore and model.training:
        return model
    if do:
        if is_main():
            dprint(f"Compiling {model.__class__.__name__} with {backend}...")
        model = torch.compile(model, backend=backend, **kwargs)
        if is_main():
            dprint(f"Done compiling {model.__class__.__name__}")
        return model
    else:
        return model


def make_compile_model(backend: Optional[str], ignore: set[str] = set()):
    return functools.partial(compile_model, ignore=ignore, backend=backend)


__ts = datetime.now()
__last_ts = __ts


def dprint(*args, reset: bool = False, **kwargs):
    global __ts, __last_ts
    if reset:
        __ts = datetime.now()

    total = datetime.now() - __ts
    delta = datetime.now() - __last_ts
    print(f"[{local_rank()}/T:{total}/D:+{delta}]", *args, **kwargs, flush=True)
    __last_ts = datetime.now()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, OptimizedModule):
        return obj._orig_mod
    if isinstance(obj, PreTrainedModel):
        return obj.config
    if isinstance(obj, PretrainedConfig):
        return obj.to_dict()
    if isinstance(obj, torch.nn.Module):
        return list(obj.state_dict().keys())
    if isinstance({}, Hashable):
        return hash(obj)
    if isinstance(obj, PushToHubMixin):
        return obj.__class__.__name__
    raise TypeError


def hash_dict(d: dict):
    return sha1(
        json.dumps(
            d, option=json.OPT_SORT_KEYS | json.OPT_NON_STR_KEYS, default=json_default
        )
    ).hexdigest()


def images(p: Path):
    return list(itertools.chain(p.glob("*.jpg"), p.glob("*.png")))


def depth_image_path(path):
    dir = path.parent / "depth"
    if not dir.exists():
        dir.mkdir()
    return dir / path.name


def image_transforms(
    size: int,
    augment: bool = True,
    to_pil: bool = False,
    normalize: bool = True,
):
    t = [
        TT.ToTensor(),
        TT.Resize(
            size,
            interpolation=TT.InterpolationMode.BICUBIC,
            antialias=True,
        ),
    ]
    if augment:
        t += [
            TT.RandomCrop(size),
            TT.RandomOrder(
                [
                    TT.ColorJitter(brightness=0.2, contrast=0.1),
                    TT.RandomHorizontalFlip(p=0.5),
                    TT.RandomAdjustSharpness(2, p=0.5),
                ]
            ),
        ]
    else:
        t += [
            TT.CenterCrop(size),
        ]
    if normalize:
        t += [
            TT.Normalize([0.5], [0.5]),
        ]
    if to_pil:
        t += [
            TT.ToPILImage(),
        ]
    return TT.Compose(t)


def depth_transforms(size: int, scale_factor: float):
    return TT.Compose(
        [
            TT.Resize(
                size // scale_factor,
                interpolation=TT.InterpolationMode.BILINEAR,
            ),
            TT.ToTensor(),
        ]
    )
