import inspect
from typing import Type

import torch
from accelerate.utils import convert_outputs_to_fp32

from dreambooth.train.shared import compile_model, hash_dict


class CompiledModelsRegistryMeta(type):
    models: dict[tuple[Type[torch.nn.Module], str], torch.nn.Module]

    def compile(self, model: torch.nn.Module, **kwargs):
        if not kwargs.get("do", True):
            return model
        model.forward = torch.autocast(device_type="cuda", dtype=torch.bfloat16)(
            model.forward
        )
        model.forward = convert_outputs_to_fp32(model.forward).__call__
        model = compile_model(model, **kwargs)
        return model.to("cuda")

    def _is_inference(self):
        for frame in inspect.stack():
            if frame.function == "decorate_context":
                try:
                    return frame.frame.f_locals["ctx_factory"].__self__.mode
                except Exception:
                    continue
        return False

    def get(self, klass: Type[torch.nn.Module], *args, reset: bool = False, **kwargs):
        key = (
            klass,
            hash_dict({**kwargs, "args": args, "is_inference": self._is_inference()}),
        )
        do_compile = kwargs.pop("compile", isinstance(klass, torch.nn.Module))
        if key not in self.models:
            load = klass.from_pretrained if hasattr(klass, "from_pretrained") else klass
            self.models[key] = self.compile(load(*args, **kwargs), do=do_compile)
        elif reset:
            fresh = klass.from_pretrained(*args, **kwargs)
            fresh.state_dict(self.models[key].state_dict())
        return self.models[key]


class CompiledModelsRegistry(metaclass=CompiledModelsRegistryMeta):
    models = {}
