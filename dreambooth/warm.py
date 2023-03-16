import os
from pathlib import Path

from accelerate.commands.launch import launch_command, launch_command_parser

from dreambooth.train.train import standalone_params


def main():
    os.environ["DREAMBOOTH_ID"] = "test"
    os.environ["WARM"] = "1"

    standalone_params(True)

    parser = launch_command_parser()
    args = parser.parse_args([str(Path(__file__).parent / "train" / "train.py")])
    launch_command(args)
