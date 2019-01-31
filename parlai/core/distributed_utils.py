#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Useful utilities for training in distributed mode.
"""

import builtins
import pickle
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
        raise ValueError(
            'Distributed mode only makes sense when using GPUs.'
        )

    if opt.get('numthreads', 1) != 1:
        raise ValueError(
            '--numthreads must be 1 for distributed training.'
        )

    if 'train:stream' in opt['datatype'] or 'ordered' in opt['datatype']:
        raise ValueError(
            "You should not combine ordered streaming with distributed training "
            "because all workers will have exactly the same minibatches, "
            "defeating the purpose."
        )


def is_distributed():
    """
    Returns True if we are in distributed mode.
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
    Returns False if we are a secondary worker. Returns True if we are either
    (1) not in distributed mode (2) or are the primary (rank 0) worker.
    """
    return not is_distributed() or dist.get_rank() == 0


def override_print(suppress=False, prefix=None):
    """
    Overrides the builtin print, to either mute or annotate the output with a
    given prefix.

    Recommended usage is to call this with suppress=True for all non-primary workers,
    or call with with a prefix of rank on all workers.

    :param bool suppress: if true, all future print statements are noops.
    :param str prefix: if not None, this string is prefixed to all future print
        statements.
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

    builtins.print = new_print


def all_gather_list(data, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    :param data: data from the local worker to be gathered on other workers
    :param int max_size: maximum size of the data to be gathered across workers

    :returns: a list containing [data1, data2, ...] of all workers
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
        not hasattr(all_gather_list, '_buffer') or
        all_gather_list._buffer.numel() < buffer_size
    ):
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)

    buffer = all_gather_list._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255 * 256

    buffer_rank = buffer[rank * max_size: (rank + 1) * max_size]
    buffer_rank[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_rank[1] = enc_size % 255
    buffer_rank[2: enc_size + 2] = torch.ByteTensor(list(enc))

    dist.all_reduce(buffer)

    result = []
    for i in range(world_size):
        out_buffer = buffer[i * max_size: (i + 1) * max_size]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()
        if size > 0:
            try:
                result.append(pickle.loads(bytes(out_buffer[2:size+2].tolist())))
            except pickle.UnpicklingError:
                raise RuntimeError(
                    'There was an unpickling error in all_gather_list. This likely '
                    'means your workers got out of syncronization (e.g. one is '
                    'expecting to sync and another is not.)'
                )

    return result


def sync_object(data, max_size=16384):
    """
    Syncs an object among all workers, overriding everyone's version with the
    primary worker's. Data must be pickleable.
    """
    if not is_distributed():
        return data

    # prepare the buffer
    if (not hasattr(sync_object, '_buffer') or sync_object._buffer.numel() < max_size):
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
        buffer[2: enc_size + 2] = torch.ByteTensor(list(enc))

    dist.broadcast(buffer, 0)

    if not is_primary_worker():
        # deserialize the data
        enc_size = buffer[0].item() * 255 + buffer[1].item()
        try:
            data = pickle.loads(bytes(buffer[2: enc_size + 2].tolist()))
        except pickle.UnpicklingError:
            raise RuntimeError(
                'There was an unpickling error in sync_object. This likely '
                'means your workers got out of syncronization (e.g. one is '
                'expecting to sync and another is not.)'
            )

    return data


class DistributionConcatenation2D(torch.nn.Module):
    """
        If training is not distributed, alors it is equivalent to identity.

        Otherwise:
        Provides the result of the concatenation of 2D tensors of the same size,
        but computed on several GPUs. REQUIRES THEM TO BE THE SAME SIZE.

        forward time:
            GPU1 produces x=[[a, b], [c, d]]
            GPU2 produces x=[[e, f], [g, h]]

            y = DistributionConcatenation2D(x)
            On GPU1 y = [[a, b], [c, d], [e, f], [g, h]]
            On GPU2 y = [[a, b], [c, d], [e, f], [g, h]]

        backward_time:
            GPU1 computed grad_y = [[s1, s2], [t1, t2], [u1, u2], [v1, v2]]
            GPU1 computed grad_y = [[w1, w2], [x1, x2], [y1, y2], [z1, z2]]

            On GPU1 grad_x = [[s1 + w1, s2 + w2], [t1 + x1, t2 + x2]]
            On GPU2 grad_x = [[u1 + y1, u2 + y2], [v1 + z1, v2 + z2]]

        If the keep_self_first option is true, then at forward time:
            GPU1 produces x=[[a, b], [c, d]]
            GPU2 produces x=[[e, f], [g, h]]

            y = DistributionConcatenation2D(x)
            On GPU1 y = [[a, b], [c, d], [e, f], [g, h]]
            On GPU2 y = [[e, f], [g, h], [a, b], [c, d]]

    """

    def __init__(self, keep_self_first=False):
        """ :param keep_self_first: if True, this module will
                maintain its input (x) at the beginning of the output
        """
        self.keep_self_first = keep_self_first
        self.rank = dist.get_rank()
        self.world_size = num_workers()
        super(DistributionConcatenation2D, self).__init__()

    def forward(self, x):
        if not is_distributed():
            return x
        y = DistributionConcatenation2DFunction.apply(x,
            torch.tensor(self.rank),
            torch.tensor(self.world_size))
        bsz = x.size(0)
        if self.keep_self_first:
            y = torch.cat([
                    y[self.rank * bsz: (self.rank + 1) * bsz],
                    y[0 : self.rank * bsz],
                    y[(self.rank + 1) * bsz :]
                ], 0)
            y = y .contiguous()
        return y

class DistributionConcatenation2DFunction(torch.autograd.Function):
    """ Actual function behind DistributionConcatenation2D. See comments of the
        module.
    """

    @staticmethod
    def forward(ctx, x, rank, world_size):
        ctx.save_for_backward(rank, world_size)
        gather_list = [x.new_zeros(x.size(0), x.size(1)) for _ in range(world_size)]
        print(x.size())
        dist.all_gather(gather_list, x)
        y = torch.cat(gather_list, 0).contiguous()
        return y

    @staticmethod
    def backward(ctx, grad_y):
        rank = ctx.saved_tensors[0]
        world_size = ctx.saved_tensors[1]

        # gather all the gradients that have been computed everywhere
        gather_list = [grad_y.new_zeros(grad_y.size(0), grad_y.size(1))
                       for _ in range(world_size)]
        dist.all_gather(gather_list, grad_y.contiguous())

        # collect them as a tensor instead of a list
        bsz = grad_y.size(0) / world_size
        gather_tensor = torch.cat([g.unsqueeze(0) for g in gather_list], 0)

        # restrict to x and returns gradient
        gather_tensor_x = gather_tensor[:, rank * bsz:(rank + 1) * bsz, :]
        grad_x = torch.sum(gather_tensor_x, 0)
        return grad_x, None, None
