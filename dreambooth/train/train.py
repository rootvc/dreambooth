import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict, Union

from sagemaker_training import environment

from dreambooth.train.utils import HyperParams, get_model, get_params


class Params(TypedDict):
    instance_path: Path
    params: HyperParams


def sagemaker_params(env: environment.Environment) -> Params:
    train_data = Path(env.channel_input_dirs["train"])
    model_data = Path(env.channel_input_dirs["model"])

    params = get_params()
    model_file = model_data / Path(params.model.name).with_suffix(".tpxz").name

    model_dir = tempfile.mkdtemp()
    with tempfile.NamedTemporaryFile() as f:
        subprocess.check_call(
            [
                "pixz",
                "-d",
                model_file,
                f.name,
            ]
        )
        shutil.unpack_archive(f.name, model_dir, format="tar")
    print("Model directory:", model_dir)
    subprocess.run(["ls", "-l", model_dir])

    params.model.name = model_dir

    return {"instance_path": train_data, "params": params}


def standalone_params(env: environment.Environment):
    params = get_params()
    example_data = Path(__file__).parent.parent / "data" / "example"
    return {"instance_path": example_data, "params": params}


def main():
    env = environment.Environment()
    if env.channel_input_dirs:
        params = sagemaker_params(env)
    else:
        params = standalone_params(env)

    model = get_model(**params)
    model.train()

    shutil.copytree(model.output_dir, env.model_dir)


if __name__ == "__main__":
    main()
