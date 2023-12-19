
# RootVC Dreambooth

## Setup Instructions

1. Clone the repo, and pip install everything in `requirements.txt`, `requirements-dev.txt`, and `environment.txt`.
2. Set the printer device name (use `lpstat -a` to find) [here](https://github.com/rootvc/dreambooth/blob/null/print-server/print_server/api/index.py#L9).
3. Start the print server by `cd`ing into `print-server` and running `python -m print_server`
4. Set the Photobooth to POST photos to https://dreambooth.vercel.app/api/media


## Install

`pip install -r requirements.txt -r requirements-dev.txt -r environment.txt`

`pip install modal loguru controlnet_aux cloudpathlib "pydantic<2"`

`git clone https://github.com/ZhaoJ9014/face.evoLVe.PyTorch.git /Users/yasyf/.virtualenvs/dreambooth-serverless/lib/python3.11/site-packages/face_evolve`

`pip install git+https://github.com/tencent-ailab/IP-Adapter.git`

`modal setup --profile rootvc`

## Deploy

Docker -> Depot -> DockerHub

`./scripts/build_image_local.sh`

DockerHub -> Modal

`MODAL_PROFILE=rootvc modal deploy one_shot::stub`
