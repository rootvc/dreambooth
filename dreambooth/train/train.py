import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import TypedDict

from sagemaker_training import environment

from dreambooth.params import Class
from dreambooth.train.utils import HyperParams, get_model, get_params

warnings.filterwarnings("ignore", module=r"(.*)\._functorch\.(.*)")


class Params(TypedDict):
    instance_path: Path
    params: HyperParams


def _unpack_model(env: environment.Environment, name: str):
    model_data = Path(env.channel_input_dirs["model"])
    model_dir = tempfile.mkdtemp()

    model_file = model_data / Path(name).with_suffix(".tpxz")
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
    return model_dir


def sagemaker_params(env: environment.Environment) -> Params:
    train_data = Path(env.channel_input_dirs["train"])
    prior_data = Path(env.channel_input_dirs["prior"])
    params = get_params()

    params.model.name = _unpack_model(env, params.model.name)
    if params.model.vae:
        params.model.vae = _unpack_model(env, params.model.vae)
    params.prior_class = Class(prompt=params.prior_prompt, data=prior_data)

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

    shutil.copytree(model.output_dir, env.model_dir, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
