# syntax=docker/dockerfile:1.4

FROM rootventures/train-dreambooth:latest

WORKDIR /dreambooth

ADD --link dreambooth/param/ dreambooth/param/
ADD --link dreambooth/download.py dreambooth/download.py

RUN python3 -m dreambooth.download

ADD --link . .
