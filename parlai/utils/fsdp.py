#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for FullyShardedDataParallel.
"""

import contextlib
import torch.nn
from parlai.utils.distributed import is_distributed, get_dist_group

try:
    from fairscale.nn.wrap.auto_wrap import wrap
    from fairscale.nn.wrap.auto_wrap import enable_wrap as fairscale_enable_wrap
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

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

    # zero3 not supported at this time. Throw an exception
    if opt['ddp_backend'] == 'zero3':
        raise NotImplementedError(
            '--ddp-backend zero3 is not supported at this time. For details, see '
            'https://github.com/facebookresearch/ParlAI/issues/3753.'
        )

    reshard_after_forward = opt['ddp_backend'] == 'zero3'
    compute_dtype = torch.float16 if opt['fp16'] else torch.float32
    mixed_precision = opt['fp16'] and opt['fp16_impl'] == 'safe'
    fsdp_args = dict(
        reshard_after_forward=reshard_after_forward,
        mixed_precision=mixed_precision,
        compute_dtype=compute_dtype,
        state_dict_device=torch.device('cpu'),
        flatten_parameters=True,
        process_group=get_dist_group(),
    )
    with fairscale_enable_wrap(wrapper_cls=FSDP, **fsdp_args):
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
