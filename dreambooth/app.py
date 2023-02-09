import torch
from accelerate.logging import get_logger
from diffusers.utils import check_min_version, is_wandb_available
from transformers import pipeline

check_min_version("0.13.0.dev0")
logger = get_logger(__name__)


def get_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("fill-mask", model="bert-base-uncased", device=device)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get("prompt", None)
    if prompt == None:
        return {"message": "No prompt provided"}

    # Run the model
    result = model(prompt)

    # Return the results as a dictionary
    return result
