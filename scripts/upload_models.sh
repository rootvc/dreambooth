#!/bin/bash
set -euxo pipefail

export OC_CAUSE=1
pip install omegaconf

python3 -m dreambooth.download
