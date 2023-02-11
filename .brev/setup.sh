#!/bin/bash

set -eo pipefail

echo 'export PATH=~/.local/bin:$PATH' >>~/.bashrc
echo 'export ACCELERATE_LOG_LEVEL=debug' >>~/.bashrc
