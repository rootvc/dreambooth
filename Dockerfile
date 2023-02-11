FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /

RUN apt-get update && apt-get install -y git

RUN pip3 install --upgrade pip

WORKDIR /dreambooth

ADD requirements.txt requirements.txt
# RUN pip3 install -U --pre -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu118

ADD . .

# RUN python3 dreambooth/download.py

EXPOSE 8000
CMD python3 -u server.py
