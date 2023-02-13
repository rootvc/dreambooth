import asyncio

from sagemaker.estimator import Estimator
from trainml.trainml import TrainML


class TrainJob:
    def __init__(self, id: str):
        self.trainml = TrainML()
        self.id = id

    async def run(self, instance: str = "ml.p4d.24xlarge", dtype: str = "bf16"):
        estimator = Estimator(
            image_uri="630351220487.dkr.ecr.us-west-2.amazonaws.com/train-dreambooth-sagemaker:latest",
            role="SageMakerRole",
            use_spot_instances=True,
            max_run=60 * 60 * 1,
            train_instance_count=1,
            train_instance_type=instance,
            environment={
                "INSTANCE_TYPE": instance,
                "ACCELERATE_MIXED_PRECISION": dtype,
            },
        )
        estimator.fit({"train": f"s3://dreambooth-datasets/{self.id}"})


if __name__ == "__main__":
    asyncio.run(
        TrainJob("test").run(
            instance="ml.p3.2xlarge",
            dtype="fp16",
        )
    )
