import asyncio
import os
import tempfile

from cloudpathlib import CloudPath
from sagemaker.estimator import Estimator
from sagemaker.inputs import FileSystemInput, TrainingInput

from dreambooth.download import download_model
from dreambooth.params import HyperParams


class TrainJob:
    BUCKET = "s3://rootvc-photobooth"
    MAX_RUN = 60 * 60 * 1
    MAX_WAIT = 60 * 10

    def __init__(self, id: str):
        self.id = id
        self.check_model()

    @property
    def model_name(self):
        return HyperParams().model.name

    def check_model(self):
        bucket = CloudPath(self.BUCKET)
        model_path = bucket / "models" / self.model_name

        if model_path.is_dir():
            print("Model already uploaded!")
            return
        else:
            print("Uploading model...")

        model = download_model()
        with tempfile.TemporaryDirectory() as dir:
            model.save_pretrained(dir)
            model_path.upload_from(dir)

    async def run(self, instance: str = "ml.p4d.24xlarge", dtype: str = "bf16"):
        estimator = Estimator(
            image_uri="630351220487.dkr.ecr.us-west-2.amazonaws.com/train-dreambooth-sagemaker:latest",
            role="SageMakerRole",
            use_spot_instances=True,
            max_run=self.MAX_RUN,
            max_wait=self.MAX_RUN + self.MAX_WAIT,
            instance_count=1,
            instance_type=instance,
            environment={
                "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
                "INSTANCE_TYPE": instance,
                "ACCELERATE_MIXED_PRECISION": dtype,
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
                    s3_data=f"{self.BUCKET}/dataset/{self.id}",
                    input_mode="FastFile",
                ),
                "model": FileSystemInput(
                    file_system_id="fs-0cbeda3084aca5585",
                    file_system_type="FSxLustre",
                    directory_path=f"/teld3bev/models/{self.model_name}",
                    file_system_access_mode="ro",
                ),
            }
        )


if __name__ == "__main__":
    asyncio.run(
        TrainJob("test").run(
            instance="ml.p3.2xlarge",
            dtype="fp16",
        )
    )
