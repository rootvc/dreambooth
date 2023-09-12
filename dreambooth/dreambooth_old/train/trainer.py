import asyncio

from dreambooth_old.train.trainers.runpod import RunpodTrainer


def run(id: str):
    asyncio.run(RunpodTrainer(id).run_and_wait())


if __name__ == "__main__":
    run("test")
