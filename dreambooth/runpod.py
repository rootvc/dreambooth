import os
import traceback
import warnings

import runpod.serverless

from dreambooth.train.train import main as train


def prepare():
    print("Warming...")

    os.environ["WARM"] = "1"
    os.environ["DREAMBOOTH_ID"] = "test"
    os.environ["WANDB_MODE"] = "disabled"

    try:
        # train()
        pass
    finally:
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
    try:
        prepare()
    except Exception:
        traceback.print_exc()
        warnings.warn("Warmup failed. Requests will likely fail too.")
    runpod.serverless.start({"handler": run})


if __name__ == "__main__":
    main()
