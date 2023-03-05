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
    from peft.tuners.lora import LoraModel
    from torch._dynamo.eval_frame import OptimizedModule

    pipelines.pipeline_utils.LOADABLE_CLASSES["transformers"]["OptimizedModule"] = [
        "save_pretrained",
        "from_pretrained",
    ]
    setattr(importlib.import_module("transformers"), "OptimizedModule", OptimizedModule)

    pipelines.pipeline_utils.LOADABLE_CLASSES["transformers"]["LoraModel"] = [
        "save_pretrained",
        "from_pretrained",
    ]
    setattr(importlib.import_module("transformers"), "OptimizedModule", LoraModel)

    yield
    del pipelines.pipeline_utils.LOADABLE_CLASSES["transformers"]["OptimizedModule"]
