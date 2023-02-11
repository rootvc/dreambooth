FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /

ARG WANDB_API_KEY

ENV ACCELERATE_MIXED_PRECISION fp16
ENV WANDB_API_KEY $WANDB_API_KEY
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib"

RUN apt-get update && apt-get install -y git

RUN pip3 install --upgrade pip

WORKDIR /dreambooth

ADD requirements.txt requirements.txt
RUN pip3 install -U --pre -r requirements.txt

ADD dreambooth/params.py dreambooth/params.py
ADD dreambooth/download.py dreambooth/download.py

RUN python3 -m dreambooth.download

ADD . .

RUN mkdir -p ~/.cache/huggingface/accelerate && \
  mv dreambooth/data/accelerate_config.yml ~/.cache/huggingface/accelerate/default_config.yaml

EXPOSE 8000
CMD python3 -u server.py
