#!/bin/bash
set -euxo pipefail

export DOCKER_BUILDKIT=1
export DEPOT_PROJECT=dww35s1t1q

build_push() {
  DOCKERFILE="dockerfiles/$1"
  IMAGE_NAME="rootventures/$2"

  depot build \
    --project $DEPOT_PROJECT \
    --pull \
    --platform linux/amd64 \
    -o type=registry,oci-mediatypes=true,compression=gzip,force-compression=true \
    -f "$DOCKERFILE" . \
    -t "$IMAGE_NAME:latest" \
    -t "$IMAGE_NAME:main"
}

build_push Dockerfile.pytorch pytorch min
build_push Dockerfile.train train-dreambooth min
build_push Dockerfile.modal train-dreambooth-modal min
# build_push Dockerfile.runpod train-dreambooth-runpod min
