import functools
import importlib
import itertools
import warnings
from contextlib import contextmanager
from typing import Callable, TypeVar

T = TypeVar("T")


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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

    for lib in LIBS:
        delattr(pipelines, lib)
