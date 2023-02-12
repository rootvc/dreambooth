import os
from pathlib import Path

from dreambooth.train.utils import get_model


def main():
    data = Path(os.environ["TRAINML_DATA_PATH"])
    images = [f.read_bytes() for f in data.glob("*.jpg")]
    model = get_model(instance_images=images)
    model.train()


if __name__ == "__main__":
    main()
