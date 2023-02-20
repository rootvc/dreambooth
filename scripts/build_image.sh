#!/bin/bash

export DOCKER_BUILDKIT=1

docker build -f dockerfiles/Dockerfile.train . -t train-dreambooth
