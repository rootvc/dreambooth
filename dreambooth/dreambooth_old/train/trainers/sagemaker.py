import asyncio
import itertools
import logging
import subprocess
import tempfile
import time
from enum import Enum, auto
from functools import lru_cache, partial
from operator import attrgetter
from typing import Callable, Iterable, Optional, Type, Union

import boto3
from cloudpathlib import CloudPath
from diffusers import AutoencoderKL, StableDiffusionPipeline
from pydantic import BaseModel
from sagemaker.estimator import Estimator
from sagemaker.inputs import FileSystemInput, TrainingInput
from transformers import CLIPModel, PreTrainedModel

from dreambooth_old.download import download, download_test_models
from dreambooth_old.params import HyperParams
from dreambooth_old.train.shared import hash_bytes

from .base import BaseTrainer

try:
    from rich import print
except ImportError:
    pass


class IntanceConfig(BaseModel):
    instance: str

    class Config:
        frozen = True


class InstanceOptimizer(Enum):
    COST = auto()
    TIME = auto()
    BALANCE = auto()


class SagemakerTrainer(BaseTrainer):
    BUCKET_ALIAS = (
        "s3://rootvc-photobooth-ac-ijsnw37rpoofofjypoen8a9oarxgyusw2b-s3alias"
    )

    DEFAULT_MULTI_INSTANCES = [
        IntanceConfig(instance="ml.g5.48xlarge"),
        IntanceConfig(instance="ml.p3dn.24xlarge"),
        IntanceConfig(instance="ml.p4d.24xlarge"),
        IntanceConfig(instance="ml.p4de.24xlarge"),
    ]

    DEFAULT_BUDGET_MULTI_INSTANCES = [
        IntanceConfig(instance="ml.g5.12xlarge"),
        IntanceConfig(instance="ml.g5.24xlarge"),
        IntanceConfig(instance="ml.p3.8xlarge"),
        IntanceConfig(instance="ml.p3.16xlarge"),
        IntanceConfig(instance="ml.g4dn.12xlarge"),
    ]

    DEFAULT_MODERN_SINGLE_INSTANCES = [
        IntanceConfig(instance="ml.g5.8xlarge"),
        IntanceConfig(instance="ml.g5.4xlarge"),
        IntanceConfig(instance="ml.g5.2xlarge"),
        IntanceConfig(instance="ml.g5.16xlarge"),
        IntanceConfig(instance="ml.g5.xlarge"),
    ]

    DEFAULT_BUDGET_INSTANCES = [
        IntanceConfig(instance="ml.g4dn.8xlarge"),
        IntanceConfig(instance="ml.p3.2xlarge"),
        IntanceConfig(instance="ml.g4dn.16xlarge"),
    ]

    estimator: Estimator

    def __init__(
        self, id: str, optimizer: InstanceOptimizer = InstanceOptimizer.BALANCE
    ):
        super().__init__(id)
        self.instance_optimizer = optimizer
        self.check_output()
        self.check_priors()
        self.check_model(StableDiffusionPipeline, **self.model_params)
        if self.vae_params:
            self.check_model(AutoencoderKL, **self.vae_params)
        self.check_model(
            CLIPModel, HyperParams().test_model, download_fn=download_test_models
        )

    @property
    def instance_options(self) -> Iterable[IntanceConfig]:
        instances = [
            self.DEFAULT_BUDGET_INSTANCES,
            self.DEFAULT_MODERN_SINGLE_INSTANCES,
            self.DEFAULT_BUDGET_MULTI_INSTANCES,
            self.DEFAULT_MULTI_INSTANCES,
        ]
        if self.instance_optimizer != InstanceOptimizer.COST:
            instances.remove(self.DEFAULT_BUDGET_INSTANCES)

        match self.instance_optimizer:
            case InstanceOptimizer.COST:
                return sum(
                    map(
                        partial(sorted, key=attrgetter("instance")),
                        instances,
                    ),
                    [],
                )
            case InstanceOptimizer.TIME:
                return sum(
                    map(
                        partial(sorted, reverse=True, key=attrgetter("instance")),
                        reversed(instances),
                    ),
                    [],
                )
            case InstanceOptimizer.BALANCE:
                return sum(reversed(instances), [])

    @property
    def model_params(self) -> dict:
        params = HyperParams()
        return {
            "name": params.model.name,
            "revision": params.model.revision,
            "torch_dtype": params.dtype,
        }

    @property
    def vae_params(self) -> Optional[dict]:
        params = HyperParams()
        if params.model.vae:
            return {"name": params.model.vae}

    @property
    def priors_hash(self):
        return hash_bytes(HyperParams().prior_prompt.encode())

    def job_base_name(self, config: IntanceConfig):
        return f"dreambooth-{config.instance.replace('.', '')}-{self.id}"

    @property
    def keep_alive(self):
        match self.instance_optimizer:
            case InstanceOptimizer.COST:
                return None
            case InstanceOptimizer.TIME:
                return 60 * 60
            case InstanceOptimizer.BALANCE:
                return 60 * 5

    def use_spot(self, config: IntanceConfig):
        return (
            self.instance_optimizer == InstanceOptimizer.COST
            or self.get_quota(config.instance) == 0
        )

    @lru_cache
    def get_quota(self, instance: str) -> int:
        client = boto3.client("service-quotas")
        pages = client.get_paginator("list_service_quotas").paginate(
            ServiceCode="sagemaker", PaginationConfig={"PageSize": 100}
        )
        quotas = itertools.chain.from_iterable(page["Quotas"] for page in pages)
        return next(
            (
                q["Value"]
                for q in quotas
                if q["QuotaName"] == f"{instance} for training warm pool usage"
            ),
            0,
        )

    def check_priors(self):
        bucket = CloudPath(self.BUCKET)
        priors_path = bucket / "priors" / self.priors_hash

        if priors_path.is_dir():
            print("Priors already uploaded!")
            return

        raise RuntimeError("Priors not found!")

    def check_model(
        self,
        klass: Type,
        name: str,
        download_fn: Callable[
            ..., Union[PreTrainedModel, list[PreTrainedModel]]
        ] = download,
        **kwargs,
    ):
        bucket = CloudPath(self.BUCKET)
        model_path = bucket / "models" / f"{name}.tpxz"

        if model_path.is_file():
            print(f"Model {name} already uploaded!")
            return
        else:
            print(f"Downloading model {name}...")

        models = download_fn(klass, name, **kwargs)
        if not isinstance(models, list):
            models = [models]
        with tempfile.TemporaryDirectory() as dir:
            print("Saving model to", dir)
            for model in models:
                model.save_pretrained(dir)
            with tempfile.NamedTemporaryFile() as f:
                print(f"Compressing model to {f.name}...")
                subprocess.check_call(
                    [
                        "tar",
                        "--use-compress-program",
                        "pixz",
                        "-C",
                        dir,
                        "-cvf",
                        f.name,
                        ".",
                    ]
                )
                print(f"Uploading model to {model_path}...")
                model_path.upload_from(f.name)

    def check_cache(self, config: IntanceConfig):
        bucket = CloudPath(self.BUCKET)
        path = bucket / "cache" / config.instance / ".keep"
        path.touch(exist_ok=True)
        print(f"Cache path: {path.parent}")

    def check_output(self):
        bucket = CloudPath(self.BUCKET)
        path = bucket / "output" / self.id / ".keep"
        path.touch(exist_ok=True)

    async def _run(self, config: IntanceConfig):
        self.check_cache(config)

        estimator = self.estimator = Estimator(
            image_uri="630351220487.dkr.ecr.us-west-2.amazonaws.com/train-dreambooth-sagemaker:latest",
            role="SageMakerRole",
            base_job_name=self.job_base_name(config),
            use_spot_instances=self.use_spot(config),
            max_run=self.MAX_RUN,
            max_wait=self.MAX_RUN + self.MAX_WAIT if self.use_spot(config) else None,
            instance_count=1,
            instance_type=config.instance,
            environment={
                **self.env,
                "INSTANCE_TYPE": config.instance,
            },
            subnets=["subnet-0425d46d0751e9df0"],
            security_group_ids=["sg-0edc333b71f1d600d"],
            git_config={
                "repo": "https://github.com/rootvc/dreambooth.git",
                "branch": "main",
            },
            dependencies=["dreambooth", "data/config"],
            entry_point="scripts/train_model.sh",
            container_log_level=logging.INFO,
            keep_alive_period_in_seconds=None
            if self.use_spot(config)
            else self.keep_alive,
        )
        estimator.fit(
            {
                "train": TrainingInput(
                    s3_data=f"{self.BUCKET_ALIAS}/dataset/{self.id}",
                    input_mode="FastFile",
                ),
                "prior": TrainingInput(
                    s3_data=f"{self.BUCKET_ALIAS}/priors/{self.priors_hash}",
                    input_mode="FastFile",
                ),
                "model": FileSystemInput(
                    file_system_id="fs-05309550451001f05",
                    file_system_type="FSxLustre",
                    directory_path="/i5ntrbev/models",
                    file_system_access_mode="ro",
                ),
                "output": FileSystemInput(
                    file_system_id="fs-05309550451001f05",
                    file_system_type="FSxLustre",
                    directory_path=f"/i5ntrbev/output/{self.id}",
                    file_system_access_mode="rw",
                ),
                "cache": FileSystemInput(
                    file_system_id="fs-05309550451001f05",
                    file_system_type="FSxLustre",
                    directory_path=f"/i5ntrbev/cache/{config.instance}",
                    file_system_access_mode="rw",
                ),
            },
            job_name=f"{self.job_base_name(config)}-{int(time.time())}",
            wait=False,
        )
        return estimator

    def _check(self, estimator: Estimator):
        status = estimator.latest_training_job.describe()
        transitions = status["SecondaryStatusTransitions"]
        if not transitions:
            return None
        print(transitions[-1])
        if transitions[-1]["Status"] == "Failed":
            return False
        elif "Insufficient capacity" in transitions[-1]["StatusMessage"]:
            estimator.latest_training_job.stop()
            return False
        elif "Starting" in transitions[-1]["StatusMessage"]:
            return None
        return True

    async def _wait_for_start(self, estimator: Estimator):
        while (
            estimator.latest_training_job.describe()["TrainingJobStatus"]
            == "InProgress"
        ):
            match self._check(estimator):
                case bool(x):
                    return x
                case None:
                    await asyncio.sleep(5)

    async def _wait_for_finish(self, estimator: Estimator):
        while (
            estimator.latest_training_job.describe()["TrainingJobStatus"]
            == "InProgress"
        ):
            if not self._check(estimator):
                return
            await asyncio.sleep(5)

        return estimator

    async def run(self):
        for config in self.instance_options:
            print(f"[bold red]Running {config.instance}...[/bold red]")
            try:
                estimator = await self._run(config)
            except Exception as e:
                print(e)
                continue
            if await self._wait_for_start(estimator):
                print("Success!")
                yield estimator

    async def run_and_wait(self):
        async for estimator in self.run():
            if await self._wait_for_finish(estimator):
                return estimator

    async def run_and_report(self):
        async for estimator in self.run():
            return {"name": estimator.latest_training_job.job_name}
        return {"name": None}
