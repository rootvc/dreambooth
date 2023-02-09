FROM ghcr.io/pytorch/pytorch-nightly

WORKDIR /

RUN apt-get update && apt-get install -y git

RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD dreambooth .

RUN python3 dreambooth/download.py

EXPOSE 8000
CMD python3 -u dreambooth/server.py
