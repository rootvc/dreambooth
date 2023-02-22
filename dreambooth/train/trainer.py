import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Literal

from cloudpathlib import CloudPath
from pydantic import BaseModel
from sagemaker.estimator import Estimator
from sagemaker.inputs import FileSystemInput, TrainingInput

from dreambooth.download import download_model
from dreambooth.params import HyperParams
from dreambooth.train.utils import hash_bytes


class IntanceConfig(BaseModel):
    instance: str
    dtype: Literal["bf16", "fp16"]


class TrainJob:
    BUCKET = "s3://rootvc-photobooth"
    BUCKET_ALIAS = (
        "s3://rootvc-photobooth-ac-ijsnw37rpoofofjypoen8a9oarxgyusw2b-s3alias"
    )
    MAX_RUN = 60 * 60 * 1
    MAX_WAIT = 60 * 10

    DEFAULT_MULTI_INSTANCES = [
        IntanceConfig(instance="ml.g4dn.12xlarge", dtype="fp16"),
        IntanceConfig(instance="ml.g4dn.metal", dtype="fp16"),
        IntanceConfig(instance="ml.g5.12xlarge", dtype="bf16"),
        IntanceConfig(instance="ml.g5.24xlarge", dtype="bf16"),
        IntanceConfig(instance="ml.p3.8xlarge", dtype="fp16"),
        IntanceConfig(instance="ml.g5.48xlarge", dtype="bf16"),
        IntanceConfig(instance="ml.p3.16xlarge", dtype="fp16"),
        IntanceConfig(instance="ml.p3dn.24xlarge", dtype="fp16"),
        IntanceConfig(instance="ml.p4d.24xlarge", dtype="bf16"),
    ]

    DEFAULT_INSTANCES = [
        IntanceConfig(instance="ml.g5.8xlarge", dtype="bf16"),
        IntanceConfig(instance="ml.g5.4xlarge", dtype="bf16"),
        IntanceConfig(instance="ml.g5.2xlarge", dtype="bf16"),
        IntanceConfig(instance="ml.g5.xlarge", dtype="bf16"),
    ]

    estimator: Estimator

    def __init__(self, id: str, optimize_for_cost: bool = True):
        self.id = id
        self.optimize_for_cost = optimize_for_cost
        self.check_model()
        self.check_priors()

    @property
    def instance_options(self) -> Iterable[IntanceConfig]:
        instances = self.DEFAULT_INSTANCES + self.DEFAULT_MULTI_INSTANCES
        return instances if self.optimize_for_cost else reversed(instances)

    @property
    def model_name(self):
        return HyperParams().model.name

    @property
    def priors_hash(self):
        return hash_bytes(HyperParams().prior_prompt.encode())

    def check_priors(self):
        bucket = CloudPath(self.BUCKET)
        priors_path = bucket / "priors" / self.priors_hash

        if priors_path.is_dir():
            print("Priors already uploaded!")
            return

        raise RuntimeError("Priors not found!")

    def check_model(self):
        bucket = CloudPath(self.BUCKET)
        model_path = bucket / "models" / f"{self.model_name}.tpxz"

        if model_path.is_file():
            print("Model already uploaded!")
            return
        else:
            print("Downloading model...")

        model = download_model()
        with tempfile.TemporaryDirectory() as dir:
            print("Saving model to", dir)
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

    async def _run(self, config: IntanceConfig):
        estimator = self.estimator = Estimator(
            image_uri="630351220487.dkr.ecr.us-west-2.amazonaws.com/train-dreambooth-sagemaker:latest",
            role="SageMakerRole",
            use_spot_instances=True,
            max_run=self.MAX_RUN,
            max_wait=self.MAX_RUN + self.MAX_WAIT,
            instance_count=1,
            instance_type=config.instance,
            environment={
                "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
                "INSTANCE_TYPE": config.instance,
                "ACCELERATE_MIXED_PRECISION": config.dtype,
                "NVIDIA_DISABLE_REQUIRE": "true",
            },
            subnets=["subnet-0425d46d0751e9df0"],
            security_group_ids=["sg-0edc333b71f1d600d"],
            git_config={
                "repo": "https://github.com/rootvc/dreambooth.git",
                "branch": "main",
            },
            dependencies=["dreambooth", "data/config"],
            entry_point="scripts/train_model.sh",
            container_log_level=logging.DEBUG,
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
                    file_system_id="fs-0cbeda3084aca5585",
                    file_system_type="FSxLustre",
                    directory_path=str(
                        (Path(f"/teld3bev/models") / self.model_name).parent
                    ),
                    file_system_access_mode="ro",
                ),
            },
            wait=False,
        )
        return estimator

    def _check(self, estimator: Estimator):
        status = estimator.latest_training_job.describe()
        transitions = status["SecondaryStatusTransitions"]
        if not transitions:
            return None
        print(transitions[-1])
        if "Insufficient capacity" in transitions[-1]["StatusMessage"]:
            estimator.latest_training_job.stop()
            return False
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

        return estimator

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
            print(f"Running {config.instance} {config.dtype}...")
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


def run(id: str):
    asyncio.run(TrainJob(id).run_and_wait())


if __name__ == "__main__":
    run("test")
