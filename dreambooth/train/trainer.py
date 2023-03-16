import asyncio

from dreambooth.train.trainers.runpod import RunpodTrainer


def run(id: str):
    asyncio.run(RunpodTrainer(id).run_and_wait())


if __name__ == "__main__":
    run("test")
