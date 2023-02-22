#!/bin/bash
set -euxo pipefail

export DOCKER_BUILDKIT=1

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 630351220487.dkr.ecr.us-west-2.amazonaws.com

docker buildx create --use --name builder \
  --buildkitd-flags '--oci-worker-snapshotter=stargz' \
  --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=-1 \
  --driver-opt env.BUILDKIT_STEP_LOG_MAX_SPEED=-1

docker buildx inspect --bootstrap builder

docker build \
  -o type=registry,oci-mediatypes=true,compression=estargz,force-compression=true,push=true \
  --builder builder \
  -f dockerfiles/Dockerfile.pytorch . \
  -t rootventures/pytorch:latest

docker build \
  -o type=registry,oci-mediatypes=true,compression=estargz,force-compression=true,push=true \
  --builder builder \
  -f dockerfiles/Dockerfile.train . \
  -t rootventures/train-dreambooth:latest

docker build \
  -o type=registry,oci-mediatypes=true,compression=estargz,force-compression=true,push=true \
  --builder builder \
  -f dockerfiles/Dockerfile.sagemaker . \
  -t rootventures/train-dreambooth-sagemaker:latest

docker tag rootventures/train-dreambooth-sagemaker:latest 630351220487.dkr.ecr.us-west-2.amazonaws.com/train-dreambooth-sagemaker:latest
docker push 630351220487.dkr.ecr.us-west-2.amazonaws.com/train-dreambooth-sagemaker:latest

docker build \
  -o type=registry,oci-mediatypes=true,compression=estargz,force-compression=true,push=true \
  --builder builder \
  -f dockerfiles/Dockerfile . \
  -t rootventures/dreambooth:latest
