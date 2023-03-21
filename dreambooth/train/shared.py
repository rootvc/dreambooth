import functools
import importlib
import itertools
import logging
from contextlib import contextmanager
from datetime import datetime
from inspect import isfunction
from typing import Callable, Optional, TypeVar

import torch
import torch.distributed

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


def compile_model(
    model: M,
    do: bool = True,
    ignore: set[str] = set(),
    backend: Optional[str] = "inductor",
    **kwargs,
) -> M:
    from torch._dynamo.eval_frame import OptimizedModule

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
            print(f"Compiling {model.__class__.__name__} with {backend}...")
        return torch.compile(model, backend=backend, **kwargs)  # mode="max-autotune"
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
    print(f"[{local_rank()}/T:{total}/D:+{delta}]", *args, **kwargs)
    __last_ts = datetime.now()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
