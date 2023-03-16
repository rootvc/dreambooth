import os

import runpod.serverless

from dreambooth.train.test import Tester
from dreambooth.train.train import main as train
from dreambooth.train.train import standalone_params


def prepare():
    print("Warming...")

    os.environ["WARM"] = "1"
    os.environ["DREAMBOOTH_ID"] = "test"
    os.environ["WANDB_MODE"] = "disabled"

    params = standalone_params(True)
    Tester(params["params"], None, None).clip_models()

    del os.environ["WANDB_MODE"]
    del os.environ["DREAMBOOTH_ID"]
    del os.environ["WARM"]


def run(job):
    print("Running job", job)
    for k, v in job["input"]["env"].items():
        os.environ[k] = v
    try:
        train()
        return {"status": "ok"}
    finally:
        for k in job["input"]["env"]:
            del os.environ[k]


def main():
    prepare()
    runpod.serverless.start({"handler": run})


if __name__ == "__main__":
    main()
