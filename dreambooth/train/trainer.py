import asyncio

from dreambooth.train.trainers.sagemaker import InstanceOptimizer, SagemakerTrainer


def run(id: str):
    asyncio.run(SagemakerTrainer(id, optimizer=InstanceOptimizer.TIME).run_and_wait())


if __name__ == "__main__":
    run("test")
