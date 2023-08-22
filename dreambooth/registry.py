import inspect
from typing import Type

import torch
from accelerate.utils import convert_outputs_to_fp32
from accelerate.utils.operations import ConvertOutputsToFp32
from torch._dynamo.eval_frame import OptimizedModule

from dreambooth.train.shared import Mode, dprint


class CompiledModelsRegistryMeta(type):
    models: dict[tuple[Type[torch.nn.Module], str], torch.nn.Module]

    @staticmethod
    def is_wrapped(model: torch.nn.Module):
        if not hasattr(model, "forward"):
            return True
        if isinstance(model, OptimizedModule):
            model = model._orig_mod
        return isinstance(model.forward, ConvertOutputsToFp32)

    @staticmethod
    def wrap(model: torch.nn.Module, method: str = "forward"):
        if isinstance(model, OptimizedModule):
            model = model._orig_mod
        if next(model.parameters()).dtype in (torch.float16, torch.bfloat16):
            dprint("Skipping model wrapping", model.__class__.__name__)
            return model
        forward = getattr(model, method)
        forward = torch.cuda.amp.autocast(dtype=torch.bfloat16)(forward)
        forward = convert_outputs_to_fp32(forward)
        setattr(model, method, forward)
        return model

    @staticmethod
    def unwrap(model: torch.nn.Module):
        if isinstance(model, OptimizedModule):
            model = model._orig_mod
        try:
            model.forward = model.forward.__wrapped__.model_forward
        except AttributeError:
            pass
        try:
            model.forward = model.forward.__wrapped__
        except AttributeError:
            pass
        return model

    def compile(self, model: torch.nn.Module, **kwargs):
        if not hasattr(model, "forward"):
            return model
        if kwargs.pop("wrap", False):
            model = self.wrap(model)
        if kwargs.get("do", True):
            model = torch.compile(
                model, mode="reduce-overhead", fullgraph=True, **kwargs
            )
        return model.to("cuda")

    def _is_inference(self):
        for frame in inspect.stack():
            if frame.function == "decorate_context":
                try:
                    return frame.frame.f_locals["ctx_factory"].__self__.mode
                except Exception:
                    continue
        return False

    def get(
        self,
        klass: Type[torch.nn.Module],
        *args,
        reset: bool = False,
        mode: Mode = Mode.LORA,
        **kwargs,
    ):
        key = (
            klass,
            hash_dict(
                {
                    **kwargs,
                    "args": args,
                    "mode": mode,
                    "is_inference": self._is_inference(),
                }
            ),
        )
        do_compile = kwargs.pop("compile", isinstance(klass, torch.nn.Module))
        do_wrap = kwargs.pop("wrap", False)

        dprint(f"Loading model, compile={do_compile}", klass.__name__)
        load = klass.from_pretrained if hasattr(klass, "from_pretrained") else klass

        if key not in self.models:
            dprint(f"Loading model, compile={do_compile}", klass.__name__)
            load = klass.from_pretrained if hasattr(klass, "from_pretrained") else klass
            self.models[key] = self.compile(
                load(*args, **kwargs), do=do_compile, wrap=do_wrap
            )
        elif reset:
            dprint("Resetting model", klass.__name__)
            fresh = klass.from_pretrained(*args, **kwargs)
            fresh.state_dict(self.models[key].state_dict())
        return self.models[key]


class CompiledModelsRegistry(metaclass=CompiledModelsRegistryMeta):
    models = {}
