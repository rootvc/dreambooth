import asyncio

from sagemaker.estimator import Estimator
from trainml.trainml import TrainML


class TrainJob:
    def __init__(self, id: str):
        self.trainml = TrainML()
        self.id = id

    async def run(self):
        estimator = Estimator(
            image_uri="rootventures/train-dreambooth-sagemaker:latest",
            role="SageMakerRole",
            use_spot_instances=True,
            max_run=60 * 60 * 1,
            train_instance_count=1,
            train_instance_type="p4d.24xlarge",
        )
        estimator.fit({"train": f"s3://dreambooth-datasets/{self.id}"})


if __name__ == "__main__":
    asyncio.run(TrainJob("test").run())
