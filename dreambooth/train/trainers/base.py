import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import (
    Generic,
    TypeVar,
)

from git import Repo

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

    @cached_property
    def env(self):
        repo = Repo().remotes.origin
        return {
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "WANDB_GIT_COMMIT": repo.refs.main.commit.hexsha,
            "WANDB_GIT_REMOTE_URL": repo.url,
            "WANDB_NOTES": repo.refs.main.commit.summary,
            "DREAMBOOTH_ID": self.id,
            "DREAMBOOTH_BUCKET": self.BUCKET,
        }

    @abstractmethod
    async def run(self):
        pass

    @abstractmethod
    async def run_and_wait(self):
        pass

    @abstractmethod
    async def run_and_report(self) -> T:
        pass
