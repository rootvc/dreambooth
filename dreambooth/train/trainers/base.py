from abc import ABC, abstractmethod
from typing import (
    Generic,
    TypeVar,
)

try:
    pass
except ImportError:
    pass

T = TypeVar("T")


class BaseTrainer(Generic[T], ABC):
    BUCKET = "s3://rootvc-photobooth"

    MAX_RUN = 60 * 20
    MAX_WAIT = 60 * 10

    def __init__(self, id: str) -> None:
        self.id = id

    @abstractmethod
    async def run(self):
        pass

    @abstractmethod
    async def run_and_wait(self):
        pass

    @abstractmethod
    async def run_and_report(self) -> T:
        pass
