#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Useful utilities for training in distributed mode.

Many of these functions act as wrappers which perform no-ops if code is running in non-
distributed mode.
"""

import builtins
import pickle
import contextlib

try:
    import torch.version
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def validate_params(opt):
    """
    Ensure sane combinations of command line parameters for distributed training.

    Raises exceptions if anything is wrong, otherwise returns None.
    """
    if torch.version.__version__.startswith('0.'):
        raise ImportError(
            "Please upgrade to PyTorch >=1.0; "
            "visit https://pytorch.org for instructions."
        )

    if opt.get('no_cuda', False):
        raise ValueError('Distributed mode only makes sense when using GPUs.')

    if opt.get('numthreads', 1) != 1:
        raise ValueError('--numthreads must be 1 for distributed training.')

    if 'train:stream' in opt['datatype'] or 'ordered' in opt['datatype']:
        raise ValueError(
            "You should not combine ordered streaming with distributed training "
            "because all workers will have exactly the same minibatches, "
            "defeating the purpose."
        )


def is_distributed():
    """
    Return if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()


def num_workers():
    """
    Get the total number of workers.
    """
    if not is_distributed():
        return 1
    else:
        return dist.get_world_size()


def is_primary_worker():
    """
    Determine if we are the primary (master) worker.

    Returns False if we are a secondary worker. Returns True if we are either (1) not in
    distributed mode (2) or are the primary (rank 0) worker.
    """
    return not is_distributed() or dist.get_rank() == 0


@contextlib.contextmanager
def override_print(suppress=False, prefix=None):
    """
    Context manager to override the print to suppress or modify output. Recommended
    usage is to call this with suppress=True for all non-primary workers, or call with a
    prefix of rank on all workers.

    >>> with override_print(prefix="rank{}".format(rank)):
    ...     my_computation()
    :param bool suppress:
        if true, all future print statements are noops.
    :param str prefix:
        if not None, this string is prefixed to all future print statements.
    """
    builtin_print = builtins.print

    def new_print(*args, **kwargs):
        if suppress:
            # do nothing
            return
        elif prefix:
            return builtin_print(prefix, *args, **kwargs)
        else:
            # default to normal print
            return builtin_print(*args, **kwargs)

    # override the print for now
    builtins.print = new_print
    yield
    # bring it back at the end of the context
    builtins.print = builtin_print


def all_gather_list(data, max_size=16384):
    """
    Gather arbitrary data from all nodes into a list.

    Similar to `~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    :param data:
        data from the local worker to be gathered on other workers
    :param int max_size:
        maximum size of the data to be gathered across workers

    :returns:
        a list containing [data1, data2, ...] of all workers
    """
    if not is_distributed():
        # fall back to just keeping things basic if we're not distributed
        return [data]

    # stolen shamelessly from fairseq
    # https://github.com/pytorch/fairseq/blob/c37250ab1c845919af721cd3f5c4cec2993aefe1/fairseq/distributed_utils.py#L116-L170
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    buffer_size = max_size * world_size
    if (
        not hasattr(all_gather_list, '_buffer')
        or all_gather_list._buffer.numel() < buffer_size
    ):
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)

    buffer = all_gather_list._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255 * 256

    buffer_rank = buffer[rank * max_size : (rank + 1) * max_size]
    buffer_rank[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_rank[1] = enc_size % 255
    buffer_rank[2 : enc_size + 2] = torch.ByteTensor(list(enc))

    dist.all_reduce(buffer)

    result = []
    for i in range(world_size):
        out_buffer = buffer[i * max_size : (i + 1) * max_size]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()
        if size > 0:
            try:
                result.append(pickle.loads(bytes(out_buffer[2 : size + 2].tolist())))
            except pickle.UnpicklingError:
                raise RuntimeError(
                    'There was an unpickling error in all_gather_list. This likely '
                    'means your workers got out of syncronization (e.g. one is '
                    'expecting to sync and another is not.)'
                )

    return result


def sync_object(data, max_size=16384):
    """
    Sync an object among all workers.

    All workers will return the same value for `data` when returning from this
    method, always using the primary worker's version. Useful for ensuring control
    flow decisions are made the same.

    :param object data:
        The object to synchronize. Must be pickleable.
    :param int max_size:
        The maximum size of this object in bytes. Large values than 255^2 are not
        supported.

    :return: the synchronized data
    """
    if not is_distributed():
        return data

    # prepare the buffer
    if not hasattr(sync_object, '_buffer') or sync_object._buffer.numel() < max_size:
        # cuda is safe because distributed mode is only okay with CUDA
        sync_object._buffer = torch.cuda.ByteTensor(max_size)

    buffer = sync_object._buffer

    if is_primary_worker():
        enc = pickle.dumps(data)
        enc_size = len(enc)
        if (enc_size + 2 > max_size) or (enc_size > 255 * 255):
            # can't store the size in the first 2 bytes
            raise ValueError('encoded data exceeds max_size')

        buffer[0] = enc_size // 255
        buffer[1] = enc_size % 255
        buffer[2 : enc_size + 2] = torch.ByteTensor(list(enc))

    dist.broadcast(buffer, 0)

    if not is_primary_worker():
        # deserialize the data
        enc_size = buffer[0].item() * 255 + buffer[1].item()
        try:
            data = pickle.loads(bytes(buffer[2 : enc_size + 2].tolist()))
        except pickle.UnpicklingError:
            raise RuntimeError(
                'There was an unpickling error in sync_object. This likely '
                'means your workers got out of syncronization (e.g. one is '
                'expecting to sync and another is not.)'
            )

    return data


def check_synced_parameters(model):
    """
    Check that all parameters across all workers are the same.

    Always returns True, or raises an AssertionError if they are not
    synchronized.

    :param torch.nn.Module model: A pytorch model.
    :return: True
    """
    if not is_distributed():
        # if things aren't distributed, of course things are in sync
        return True

    # compute the local norm:
    norm2 = sum((p.data ** 2).sum().float() for p in model.parameters()).item()
    all_versions = all_gather_list(norm2)
    if not all(n == norm2 for n in all_versions):
        raise AssertionError(
            "Some models parameters were out of sync. Got the following norms: {}".format(
                " ".join(str(x) for x in all_versions)
            )
        )

    return True
