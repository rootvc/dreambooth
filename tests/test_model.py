import os

os.environ["ACCELERATE_LOG_LEVEL"] = "DEBUG"


from pathlib import Path

from dreambooth.utils import get_model


def main():
    data = Path(__file__).parent.parent / "data" / "example"
    images = [f.read_bytes() for f in data.glob("*.jpg")]
    model = get_model(instance_images=images)
    model.train()


if __name__ == "__main__":
    main()
