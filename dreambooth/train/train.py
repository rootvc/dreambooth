import shutil
import tempfile
from pathlib import Path

from sagemaker_training import environment

from dreambooth.train.utils import get_model, get_params


def main():
    env = environment.Environment()
    train_data = Path(env.channel_input_dirs["train"])
    model_data = Path(env.channel_input_dirs["model"])

    model_dir = tempfile.mkdtemp()
    shutil.unpack_archive(model_data, model_dir)

    params = get_params()
    params.model.name = model_dir

    model = get_model(instance_path=train_data, params=params)
    model.train()

    shutil.copytree(model.output_dir, env.model_dir)


if __name__ == "__main__":
    main()
