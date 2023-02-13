import asyncio
import tempfile

import s3fs
from sagemaker.estimator import Estimator

from dreambooth.download import download_model
from dreambooth.params import HyperParams


class TrainJob:
    BUCKET = "s3://rootvc-photobooth"

    def __init__(self, id: str):
        self.id = id
        self.check_model()

    def check_model(self):
        params = HyperParams()
        fs = s3fs.S3FileSystem()
        path = f"{self.BUCKET}/models/{params.model.name}"

        if fs.exists(path):
            return

        model = download_model()
        with tempfile.TemporaryDirectory() as dir:
            model.save_pretrained(dir)
            fs.put(dir, path, recursive=True)

    async def run(self, instance: str = "ml.p4d.24xlarge", dtype: str = "bf16"):
        estimator = Estimator(
            image_uri="630351220487.dkr.ecr.us-west-2.amazonaws.com/train-dreambooth-sagemaker:latest",
            role="SageMakerRole",
            use_spot_instances=True,
            max_run=60 * 60 * 1,
            instance_count=1,
            instance_type=instance,
            environment={
                "INSTANCE_TYPE": instance,
                "ACCELERATE_MIXED_PRECISION": dtype,
            },
        )
        estimator.fit(
            {
                "train": f"s3://dreambooth/dataset/{self.id}",
                "model": "s3://dreambooth/models",
            }
        )


if __name__ == "__main__":
    asyncio.run(
        TrainJob("test").run(
            instance="ml.p3.2xlarge",
            dtype="fp16",
        )
    )
