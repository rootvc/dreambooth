import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from textwrap import dedent
from typing import AsyncGenerator, cast

import asyncssh
from jlclient import jarvisclient
from jlclient.jarvisclient import Instance, User

from dreambooth_old.train.trainers.base import BaseTrainer

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

    async def create_instance(self) -> Instance:
        instance = cast(
            Instance,
            Instance.create(
                gpu_type="A100",
                num_gpus=4,
                hdd=80,
                framework_id=FrameworkId.BYOC.value,
                name=self.INSTANCE_NAME,
                image="rootventures/train-dreambooth-standalone:latest",
                is_reserved=False,
            ),
        )
        try:
            await self._exec(instance, "/root/setup.sh")
        except asyncssh.Error:
            instance.pause()
            raise
        return instance

    @lru_cache
    async def _instance(self) -> Instance:
        try:
            instance = self.find_instance()
            if instance.status != "Running":
                instance.resume()
            return instance
        except StopIteration:
            return await self.create_instance()

    @asynccontextmanager
    async def instance(self) -> AsyncGenerator[Instance, None]:
        instance = await self._instance()
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
                **self.env,
            }
        )

    async def _exec(self, instance: Instance, command: str):
        async with asyncssh.connect(
            instance.ssh_str, known_hosts=None, client_keys=[self.client_key]
        ) as conn:
            return await conn.run(command, check=True, timeout=self.MAX_RUN)

    async def run(self):
        async with self.instance() as instance:
            return instance

    async def run_and_wait(self):
        async with self.instance() as instance:
            await self._exec(instance, "/dreambooth/scripts/train/run.sh")

    async def run_and_report(self):
        return (await self.run()).ssh_str
