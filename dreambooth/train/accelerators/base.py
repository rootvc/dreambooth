from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Optional,
    TypeVar,
    Union,
    overload,
)

import torch
from accelerate.tracking import GeneralTracker
from typing_extensions import TypeVarTuple, Unpack

from dreambooth.params import HyperParams

T = TypeVar("T")
Ts = TypeVarTuple("Ts")


class BaseAccelerator(ABC):
    trackers: Iterable[GeneralTracker]

    def __init__(self, params: HyperParams) -> None:
        super().__init__()
        self._params = params

    @overload
    @abstractmethod
    def prepare(self, model: T) -> T:
        pass

    @overload
    @abstractmethod
    def prepare(self, model: T, *models: Unpack[Ts]) -> tuple[T, Unpack[Ts]]:
        pass

    @abstractmethod
    def prepare(self, model: T, *models: Unpack[Ts]) -> Union[T, tuple[T, Unpack[Ts]]]:
        pass

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        yield

    @abstractmethod
    def init_trackers(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def is_main_process(self) -> bool:
        pass

    @property
    @abstractmethod
    def optimizer_step_was_skipped(self) -> bool:
        pass

    @property
    @abstractmethod
    def sync_gradients(self) -> bool:
        pass

    @abstractmethod
    def wait_for_everyone(self):
        pass

    @abstractmethod
    def optimizer(self, params: list[dict], **kwargs) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def clip_grad_norm_(self, parameters: Iterable, max_norm: float):
        pass

    @abstractmethod
    def backward(self, loss, **kwargs):
        pass

    @abstractmethod
    def unwrap_model(self, model: T, keep_fp32_wrapper: bool = True) -> T:
        pass

    @abstractmethod
    def log(
        self, values: dict, step: Optional[int] = None, log_kwargs: Optional[dict] = {}
    ):
        pass

    @abstractmethod
    def get_state_dict(self, model, unwrap=True):
        pass

    @abstractmethod
    def end_training(self):
        pass

    @abstractmethod
    def save(self, obj, f):
        pass
