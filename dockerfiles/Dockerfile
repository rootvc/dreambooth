FROM rootventures/train-dreambooth:latest

WORKDIR /dreambooth

ADD dreambooth/params.py dreambooth/params.py
ADD dreambooth/download.py dreambooth/download.py

RUN python3 -m dreambooth.download

ADD . .
