#!/bin/bash

set -eo pipefail

export PATH=~/.local/bin:$PATH
export ACCELERATE_LOG_LEVEL=debug

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11.8

cd dreambooth
pip install -r requirements.txt
