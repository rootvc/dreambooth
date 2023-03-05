import functools
import importlib
import itertools
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

    PATCHES = [
        ("diffusers", "peft.tuners.lora", "LoraModel"),
        ("transformers", "peft.tuners.lora", "LoraModel"),
        ("transformers", "torch._dynamo.eval_frame", "OptimizedModule"),
    ]

    for (lib, mod, klass) in PATCHES:
        pipelines.pipeline_utils.LOADABLE_CLASSES[lib][klass] = [
            "save_pretrained",
            "from_pretrained",
        ]
        setattr(
            importlib.import_module(lib),
            klass,
            getattr(importlib.import_module(mod), klass),
        )

    yield
    for (lib, _, klass) in PATCHES:
        del pipelines.pipeline_utils.LOADABLE_CLASSES[lib][klass]
