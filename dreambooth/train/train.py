import shutil
import subprocess
import tempfile
from pathlib import Path

from sagemaker_training import environment

from dreambooth.train.utils import get_model, get_params


def main():
    env = environment.Environment()
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
    params.loading_workers = 1

    model = get_model(instance_path=train_data, params=params)
    model.train()

    shutil.copytree(model.output_dir, env.model_dir)


if __name__ == "__main__":
    main()
