import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from cloudpathlib import CloudPath
from pydantic import BaseModel
from sagemaker.estimator import Estimator
from sagemaker.inputs import FileSystemInput, TrainingInput

from dreambooth.download import download_model
from dreambooth.params import HyperParams


class IntanceConfig(BaseModel):
    instance: str
    dtype: Literal["bf16", "fp16", "fp32"]


class TrainJob:
    BUCKET = "s3://rootvc-photobooth"
    BUCKET_ALIAS = (
        "s3://rootvc-photobooth-ac-ijsnw37rpoofofjypoen8a9oarxgyusw2b-s3alias"
    )
    MAX_RUN = 60 * 60 * 1
    MAX_WAIT = 60 * 10

    estimator: Estimator

    def __init__(self, id: str):
        self.id = id
        self.check_model()

    @property
    def model_name(self):
        return HyperParams().model.name

    def check_model(self):
        bucket = CloudPath(self.BUCKET)
        model_path = bucket / "models" / f"{self.model_name}.xz"

        if model_path.is_file():
            print("Model already uploaded!")
            return
        else:
            print("Downloading model...")

        model = download_model()
        with tempfile.TemporaryDirectory() as dir:
            print("Saving model to", dir)
            model.save_pretrained(dir)
            print("Compressing model...")
            os.environ["XZ_DEFAULTS"] = "-T 0"
            file = shutil.make_archive(str(self.model_name), "xztar", dir, dir)
            print(f"Uploading model from {file}...")
            model_path.upload_from(file)

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
            entry_point="scripts/train_model.sh",
        )
        estimator.fit(
            {
                "train": TrainingInput(
                    s3_data=f"{self.BUCKET_ALIAS}/dataset/{self.id}",
                    input_mode="FastFile",
                ),
                # "model": FileSystemInput(
                #     file_system_id="fs-0cbeda3084aca5585",
                #     file_system_type="FSxLustre",
                #     directory_path=str((Path(f"/teld3bev/models") / self.model_name).parent),
                #     file_system_access_mode="ro",
                # ),
            },
            wait=False,
        )
        while (
            estimator.latest_training_job.describe()["TrainingJobStatus"]
            == "InProgress"
        ):
            await asyncio.sleep(10)
            status = estimator.latest_training_job.describe()
            transitions = status["SecondaryStatusTransitions"]
            if not transitions:
                continue
            print(transitions[-1])
            if (
                transitions
                and "Insufficient capacity" in transitions[-1]["StatusMessage"]
            ):
                estimator.latest_training_job.stop()
                return False
        return True

    async def run(
        self,
        configs: list[IntanceConfig] = [
            IntanceConfig(instance="ml.p3.2xlarge", dtype="fp16")
        ],
    ):
        for config in configs:
            print(f"Running {config.instance} {config.dtype}...")
            success = await self._run(config)
            if success:
                print("Success!")
                return


if __name__ == "__main__":
    asyncio.run(
        TrainJob("test").run(
            [
                IntanceConfig(instance="ml.p3.2xlarge", dtype="fp16"),
                IntanceConfig(instance="ml.g5.xlarge", dtype="fp16"),
                IntanceConfig(instance="ml.g4dn.2xlarge", dtype="fp16"),
                IntanceConfig(instance="ml.g4dn.xlarge", dtype="fp16"),
                IntanceConfig(instance="ml.g4dn.4xlarge", dtype="fp16"),
                IntanceConfig(instance="ml.g4dn.16xlarge", dtype="fp16"),
            ]
        )
    )
