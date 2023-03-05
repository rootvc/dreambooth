import functools
import itertools
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
