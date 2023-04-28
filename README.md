
# RootVC Dreambooth

## Setup Instructions

1. Clone the repo, and pip install everything in `requirements.txt`, `requirements-dev.txt`, and `environment.txt`.
2. Set the printer device name (use `lpstat` to find) [here](https://github.com/rootvc/dreambooth/blob/null/print-server/print_server/api/index.py#L9).
3. Start the print server by `cd`ing into `print-server` and running `python -m print_server`
4. Set the Photobooth to POST photos to https://dreambooth.vercel.app/api/media
