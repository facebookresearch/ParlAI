#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for FullyShardedDataParallel.
"""
import contextlib
import functools
import torch
import torch.distributed
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
import torch.nn

from parlai.scripts.eval_model import Evaluator
from parlai.scripts.train_model import TrainLoop
from parlai.utils.distributed import is_distributed, get_dist_group

try:
    import torch
    import torch.distributed
    import torch.distributed.fsdp
    from torch.distributed.fsdp.wrap import (
        wrap,
        enable_wrap,
        transformer_auto_wrap_policy,
    )
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
    )

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

    def wrap(module, **kwargs):
        return module


DEFAULT_DDP_BACKEND = "ddp"


def is_fsdp(module: torch.nn.Module):
    """
    Checks whether a module is fully sharded.
    """
    return FSDP_AVAILABLE and isinstance(module, FSDP)


def should_use_fsdp(opt):
    return (
        FSDP_AVAILABLE
        and is_distributed()
        and opt.get('ddp_backend', DEFAULT_DDP_BACKEND) in ('zero2', 'zero3')
    )


@contextlib.contextmanager
def maybe_fsdp_wrap(opt):
    """
    Context manager for enabling wrapping in FullyShardedDataParallel.
    """
    if not should_use_fsdp(opt):
        # make a no-op
        yield
        return

    mixed_precision = opt['fp16'] and opt['fp16_impl'] == 'safe'

    # settings as of pytorch 1.13
    # There is a warning in pytorch 1.13 for FSDP that is unavoidable;
    # at the risk of suppressing valid warnings, just going to suppress that one.
    import warnings

    warnings.filterwarnings("ignore")

    # sharding strategy determines zero2 or zero3
    sharding_strategy = (
        ShardingStrategy.FULL_SHARD
        if opt['ddp_backend'] == 'zero3'
        else ShardingStrategy.SHARD_GRAD_OP
    )

    # mp determines how to mix precision
    if mixed_precision:
        mp_strategy = MixedPrecision(
            reduce_dtype=torch.float16,
            param_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mp_strategy = None

    # autowrap policy.
    auto_wrap_policy = None
    ignored_modules = None
    if 'hugging_face' not in opt['model']:
        from parlai.agents.transformer.modules.encoder import (
            TransformerEncoderLayer,
        )
        from parlai.agents.transformer.modules.decoder import (
            TransformerDecoderLayer,
        )

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                TransformerEncoderLayer,
                TransformerDecoderLayer,
            },
        )

    # backward prefetch; determines when to fetch the parameters during backward pass
    # set to BACKWARD_PRE to increase throughput, at the cost of memory
    backward_prefetch = BackwardPrefetch.BACKWARD_POST

    # CPU offloading; this can offload parameters to the CPU
    cpu_offload = None

    fsdp_args = dict(
        process_group=get_dist_group(),
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=backward_prefetch,
        mixed_precision=mp_strategy,
        ignored_modules=ignored_modules,
        param_init_fn=None,
        device_id=opt['gpu'],
        sync_module_states=False,  # need this for syncing the first call; specify False because we do it manually after cuda
        forward_prefetch=False,  # specify true for CPU-heavy workload
        limit_all_gathers=False,  # specifying the default here
    )
    with enable_wrap(wrapper_cls=FSDP, **fsdp_args):
        yield


def delay_halving(opt):
    """
    Check whether we should keep the model in fp32 before other setup.

    When using Zero2 or Zero3 backends with mixed precision, we need to avoid converting
    the model to fp16, as the FSDP module does this for us.

    If we are using just plain DDP or MemoryEfficient optimizers, then we want
    to call half() early.
    """

    return opt['fp16'] and should_use_fsdp(opt) and opt['fp16_impl'] == 'safe'


def should_sync_gradnorm(opt):
    """
    Indicates whether fp16 optimizer wrappers should accumulate over workers.

    FP16 overflow detection and gradient clipping both require accumulating gradients
    across all workers when using FSDP, as workers only store a fraction of the
    gradients.
    """
    return (
        FSDP_AVAILABLE
        and opt['fp16']
        and opt.get('ddp_backend', DEFAULT_DDP_BACKEND) in ('zero2', 'zero3')
    )


def fsdp_wrap(module):
    """
    Helper function for wrapping the outermost root module.
    """
    return wrap(module)


def get_state_dict(model):
    """
    Get the state dict from the model.

    When using Pytorch FSDP, we can offload to CPU.
    """

    if FSDP_AVAILABLE:
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            FullStateDictConfig,
            StateDictType,
        )

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state = model.state_dict()
    else:
        state = model.state_dict()

    return state


@contextlib.contextmanager
def fsdp_join(*args):
    with Join([*args]):
        yield


class JoinableTrainLoop(TrainLoop, Joinable):
    """
    Joinable train loop.
    """

    def __init__(self, opt):
        import parlai.utils.distributed as dist_utils

        super().__init__(opt)
        self.__device = opt['gpu']
        self.__group = dist_utils.get_dist_group()

    def __call__(self):
        """
        Join caller.

        For now, don't do anything.
        """
        Join.notify_join_context(self)

    def join_hook(self, **kwargs) -> JoinHook:
        """
        Return our fake join hook.
        """
        return TrainLoopJoinHook(self)

    @property
    def join_device(self) -> torch.device:
        return self.__device

    @property
    def join_process_group(self):
        return self.__group


class TrainLoopJoinHook(JoinHook):
    """
    Join hook for train loop.

    Adapted from https://pytorch.org/tutorials/advanced/generic_join.html
    """

    def __init__(self, train_loop: JoinableTrainLoop):
        self.train_loop = train_loop

    def main_hook(self):
        pass

    def post_hook(self, is_last_joiner: bool):
        pass


class JoinableEvaluator(Evaluator, Joinable):
    """
    Joinable Evaluator.
    """

    def __init__(self, opt):
        import parlai.utils.distributed as dist_utils

        super().__init__(opt)
        self.__device = opt['gpu']
        self.__group = dist_utils.get_dist_group()

    def __call__(self):
        """
        Join caller.

        For now, don't do anything.
        """
        Join.notify_join_context(self)

    def join_hook(self, **kwargs) -> JoinHook:
        """
        Return our fake join hook.
        """
        return EvaluatorJoinHook(self)

    @property
    def join_device(self) -> torch.device:
        return self.__device

    @property
    def join_process_group(self):
        return self.__group


class EvaluatorJoinHook(JoinHook):
    def __init__(self, evaluator: JoinableEvaluator):
        self.evaluator = evaluator

    def main_hook(self):
        pass

    def post_hook(self, is_last_joiner: bool):
        pass
