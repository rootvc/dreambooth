from contextlib import contextmanager
from typing import Generator, Optional

import colossalai
import torch
import torch.distributed
from accelerate import Accelerator as HFAccelerator
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.optimizer.zero_optimizer import ZeroOptimizer
from colossalai.nn.parallel import GeminiDDP
from colossalai.nn.parallel.utils import get_static_torch_model
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext

from dreambooth.params import HyperParams
from dreambooth.train.accelerators.base import BaseAccelerator


class ColossalAccelerator(BaseAccelerator):
    def __init__(self, params: HyperParams, **kwargs) -> None:
        super().__init__(params)
        colossalai.launch_from_torch(config={})

        self._accelerator = HFAccelerator(**kwargs)
        self._models = []
        self._optimizer = None
        self.state = {}

    @property
    def local_rank(self) -> int:
        return gpc.get_local_rank(ParallelMode.DATA)

    @property
    def world_size(self) -> int:
        return gpc.get_world_size(ParallelMode.DATA)

    @property
    def trackers(self):
        return self._accelerator.trackers

    def init_trackers(self, *args, **kwargs):
        self._accelerator.init_trackers(*args, **kwargs)

    def log(
        self, values: dict, step: Optional[int] = None, log_kwargs: Optional[dict] = {}
    ):
        self._accelerator.log(values, step=step, log_kwargs=log_kwargs)

    def end_training(self):
        self._accelerator.end_training()

    def save(self, obj, f):
        self._accelerator.save(obj, f)

    @property
    def device(self) -> torch.device:
        return get_current_device()

    @property
    def is_main_process(self) -> bool:
        return self.local_rank == 0

    @property
    def sync_gradients(self) -> bool:
        return True

    @property
    def optimizer_step_was_skipped(self) -> bool:
        return False

    def wait_for_everyone(self):
        torch.cuda.synchronize()

    def prepare(self, *models):
        for model in models:
            if isinstance(model, torch.nn.Module):
                self._models.append(model)

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        with ColoInitContext(device=self.device):
            yield

    def optimizer(self, params: dict, **kwargs) -> torch.optim.Optimizer:
        model = GeminiDDP(
            torch.nn.ModuleList(self._models),
            device=self.device,
            placement_policy="auto",
            pin_memory=True,
            search_range_mb=64,
        )
        optimizer = HybridAdam(params, **kwargs)
        self._optimizer = ZeroOptimizer(
            optimizer,
            model,
            clipping_norm=self._params.max_grad_norm,
            initial_scale=2**5,
            **kwargs
        )
        return self._optimizer

    def clip_grad_norm_(self, parameters: list, max_norm: float, norm_type: int):
        pass

    def backward(self, loss, **kwargs):
        return self._optimizer.backward(loss)

    def unwrap_model(self, model, keep_fp32_wrapper: bool = True):
        return get_static_torch_model(model)

    def get_state_dict(self, model, unwrap=True):
        return self.unwrap_model(model).state_dict()
