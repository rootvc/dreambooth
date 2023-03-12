import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum, auto
from functools import lru_cache
from textwrap import dedent
from typing import Generator, cast

import asyncssh
from jlclient import jarvisclient
from jlclient.jarvisclient import Instance, User

from dreambooth.train.trainers.base import BaseTrainer

jarvisclient.token = os.environ["JARVIS_TOKEN"]
jarvisclient.user_id = os.environ["JARVIS_USER_ID"]


class FrameworkId(Enum):
    PYTORCH = auto()
    FASTAI = auto()
    TENSORFLOW = auto()
    BYOC = auto()


@dataclass
class Params:
    env: dict[str, str]


class JarvisTrainer(BaseTrainer):
    INSTANCE_NAME = "dreambooth"

    def __init__(self, id: str) -> None:
        super().__init__(id)

    def find_instance(self):
        return next(
            cast(Instance, i)
            for i in User.get_instances()
            if i.name == self.INSTANCE_NAME
        )

    def create_instance(self) -> Instance:
        return Instance.create(
            gpu_type="A100",
            num_gpus=4,
            hdd=50,
            framework_id=FrameworkId.BYOC.value,
            name=self.INSTANCE_NAME,
            image="rootventures/train-dreambooth-sagemaker:latest",
            is_reserved=False,
            docker_username=os.environ["DOCKER_USERNAME"],
            docker_password=os.environ["DOCKER_PASSWORD"],
        )

    @lru_cache
    def _instance(self) -> Instance:
        try:
            instance = self.find_instance()
            if instance.status != "Running":
                instance.resume()
            return instance
        except StopIteration:
            return self.create_instance()

    @contextmanager
    def instance(self) -> Generator[Instance, None, None]:
        instance = self._instance()
        try:
            yield instance
        finally:
            instance.pause()

    @property
    def client_key(self) -> asyncssh.SSHKey:
        return asyncssh.import_private_key(
            dedent(
                f"""
                    -----BEGIN OPENSSH PRIVATE KEY-----
                    {os.environ["JARVIS_SSH_KEY"]}
                    -----END OPENSSH PRIVATE KEY-----
                """
            )
        )

    @property
    def params(self) -> Params:
        return Params(
            env={
                "DREAMBOOTH_ID": self.id,
                "DREAMBOOTH_BUCKET": self.BUCKET,
            }
        )

    async def run(self):
        with self.instance() as instance:
            return instance

    async def run_and_wait(self):
        with self.instance() as instance:
            async with asyncssh.connect(
                instance.ssh_str, known_hosts=None, client_keys=[self.client_key]
            ) as conn:
                await conn.run(
                    "/dreambooth/scripts/run.sh",
                    check=True,
                    timeout=self.MAX_RUN,
                    input=json.dumps(asdict(self.params)),
                    stdout=asyncssh.STDOUT,
                )

    async def run_and_report(self):
        return (await self.run()).ssh_str
