import shutil
from pathlib import Path

from sagemaker_training import environment

from dreambooth.train.utils import get_model


def main():
    env = environment.Environment()
    data = Path(env.channel_input_dirs["train"])
    images = [f.read_bytes() for f in data.glob("*.jpg")]

    model = get_model(instance_images=images)
    model.train()

    shutil.copytree(model.output_dir, env.model_dir)


if __name__ == "__main__":
    main()
