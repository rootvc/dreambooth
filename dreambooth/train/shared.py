import functools
import importlib
import itertools
import logging
from contextlib import contextmanager
from typing import Callable, TypeVar

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
        if not self.accelerator.is_main_process:
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


def is_main():
    return torch.distributed.get_rank() == 0


def compile_model(model: M, do: bool = True, ignore: set[str] = set()) -> M:
    BROKEN_COMPILE_CLASSES = set()

    if model.__class__.__name__ in BROKEN_COMPILE_CLASSES:
        return model
    elif model.__class__.__name__ in ignore and model.training:
        return model
    if do:
        if is_main():
            print(f"Compiling {model.__class__.__name__}...")
        return torch.compile(model, mode="max-autotune")
    else:
        return model


def make_compile_model(ignore: set[str]):
    return functools.partial(compile_model, ignore=ignore)
