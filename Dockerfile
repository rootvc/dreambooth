FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /

RUN apt-get update && apt-get install -y git

RUN pip3 install --upgrade pip

WORKDIR /dreambooth

ADD requirements.txt requirements.txt
RUN pip3 install -U --pre -r requirements.txt

ADD dreambooth/params.py dreambooth/params.py
ADD dreambooth/download.py dreambooth/download.py

RUN python3 dreambooth/download.py

ADD . .

EXPOSE 8000
CMD python3 -u server.py
