import torch
from accelerate import Accelerator

from dreambooth.params import HyperParams
from dreambooth.train.accelerators.base import BaseAccelerator


class HFAccelerator(Accelerator, BaseAccelerator):
    def __init__(self, params: HyperParams, **kwargs) -> None:
        BaseAccelerator.__init__(self, params)
        Accelerator.__init__(self, **kwargs)

    def optimizer(self, params: list[dict], **kwargs) -> torch.optim.Optimizer:
        try:
            if self.state.deepspeed_plugin:
                raise RuntimeError("DeepSpeed is not compatible with bitsandbytes.")

            import bitsandbytes as bnb
        except Exception as e:
            print(e)
            raise RuntimeError("Could not import bitsandbytes, using AdamW")
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = bnb.optim.AdamW8bit

        return optimizer_class(params, **kwargs)
