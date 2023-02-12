import asyncio

from sagemaker.estimator import Estimator
from trainml.trainml import TrainML


class TrainJob:
    def __init__(self, id: str):
        self.trainml = TrainML()
        self.id = id

    async def run(self):
        estimator = Estimator(
            image_uri="rootventures/train-dreambooth-sagemaker",
            role="SageMakerRole",
            train_instance_count=1,
            train_instance_type="local",
        )

        dataset = await self.trainml.datasets.create(
            name=self.id,
            source_type="aws",
            source_uri=f"s3://dreambooth-datasets/{self.id}",
        )
        await dataset.attach()
        await self.trainml.jobs.create(
            name=self.id,
            type="training",
            gpu_type="A100",
            gpu_count=4,
            disk_size=10,
            max_price=5,
            preemptible=True,
            workers=[
                "./scripts/train_model.sh",
            ],
            data={"datasets": [{"id": dataset.id, "type": "existing"}]},
            model={"git_uri": "git@github.com:rootvc/dreambooth.git"},
            environment={
                "type": "CUSTOM",
                "custom_image": "rootventures/train-dreambooth:latest",
            },
        )


if __name__ == "__main__":
    asyncio.run(TrainJob("test").run())
