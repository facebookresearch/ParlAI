#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility methods for dealing with torch code.
"""

from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
import math
from collections import namedtuple


try:
    import torch
except ImportError:
    raise ImportError('Parlai requires pytorch. Go to http://pytorch.org to install.')

import torch.optim

"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

# according to the tensor cores documentation from nvidia, the matmuls in fp16
# must all be multiples of 8 in order to get the speedup from fp16. We set this
# as a constant here for clarity and convenience.  See
# https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/ for more
# information.
FP16_PAD_SIZE = 8


def neginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    use_cuda: bool = False,
    left_padded: bool = False,
    max_len: Optional[int] = None,
    fp16friendly: bool = False,
    device: int = -1,
) -> Tuple[torch.LongTensor, List[int]]:
    """
    Create a padded matrix from an uneven list of lists.

    Returns (padded, lengths), where padded is the padded matrix, and lengths
    is a list containing the lengths of each row.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param bool sort: If True, orders by the length
    :param int pad_idx: the value to use for padding
    :param bool use_cuda: if true, places `padded` on GPU
    :param bool left_padded:
    :param int max_len: if None, the max length is the maximum item length
    :param bool fp16friendly: if True, pads the time dimension to be a multiple of 4.
    :param int device: GPU device.

    :returns: (padded, lengths) tuple
    :rtype: (Tensor[int64], list[int])
    """

    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]  # type: ignore
    # max in time dimension
    t = max(lens) if max_len is None else max_len

    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)

    if fp16friendly and (t % FP16_PAD_SIZE != 0):
        # pad to be fp16 friendly
        t += FP16_PAD_SIZE - (t % FP16_PAD_SIZE)

    if isinstance(items[0], torch.Tensor):
        # keep type of input tensors, they may already be cuda ones
        output = items[0].new(n, t)  # type: ignore
    else:
        output = torch.LongTensor(n, t)  # type: ignore
    output.fill_(pad_idx)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            # skip empty items
            continue
        if not isinstance(item, torch.Tensor):
            # put non-tensors into a tensor
            item = torch.LongTensor(item)  # type: ignore
        if left_padded:
            # place at end
            output[i, t - length :] = item
        else:
            # place at beginning
            output[i, :length] = item

    if use_cuda:
        output = output.cuda()
        if device >= 0:
            output = output.to(device)
    return output, lens


def padded_3d(
    tensors: List[torch.LongTensor],
    pad_idx: int = 0,
    use_cuda: bool = False,
    dtype: Optional[torch.dtype] = torch.long,
    fp16friendly: bool = False,
):
    """
    Make 3D padded tensor for list of lists of 1D tensors or lists.

    :param tensors:
        list of lists of 1D tensors (or lists)
    :param pad_idx:
        padding to fill tensor with
    :param use_cuda:
        whether to call cuda() before returning
    :param bool fp16friendly:
        if True, pads the final dimension to be a multiple of 8.

    :returns:
        3D tensor with the maximum dimensions of the inputs
    """
    a = len(tensors)
    b = max(len(row) for row in tensors)  # type: ignore
    c = max(len(item) for row in tensors for item in row)  # type: ignore

    # pad empty tensors
    if fp16friendly and c % FP16_PAD_SIZE != 0:
        c += FP16_PAD_SIZE - (c % FP16_PAD_SIZE)
    c = max(c, 1)

    output = torch.full((a, b, c), pad_idx, dtype=dtype)

    for i, row in enumerate(tensors):
        item: Sized
        for j, item in enumerate(row):  # type: ignore
            if len(item) == 0:
                continue
            if not isinstance(item, torch.Tensor):
                item = torch.Tensor(item, dtype=dtype)
            output[i, j, : len(item)] = item

    if use_cuda:
        output = output.cuda()

    return output


def concat_without_padding(text_idx, cand_idx, use_cuda, null_idx=0):
    """
    Concatenate two right padded tensors and move padding to the right.

    For example,
        if text_idx = [[1, 2, 3, 4, 0, 0  ]]
        and cand_idx = [[5, 6, 7, 8, 0, 0 ]]:
    Then result = (tokens, segments) where
        tokens = [[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]]
        segments = [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]
    """
    assert text_idx.size(0) == cand_idx.size(0)
    assert len(text_idx.size()) == 2
    assert len(cand_idx.size()) == 2
    segments_idx = [0, 1]
    text_idx = text_idx.cpu()
    cand_idx = cand_idx.cpu()
    cand_len = cand_idx.size(1)
    concat_len = text_idx.size(1) + cand_idx.size(1)
    tokens = text_idx.new_zeros(text_idx.size(0), concat_len) + null_idx
    segments = text_idx.new_zeros(text_idx.size(0), concat_len) + null_idx
    for i in range(len(tokens)):
        non_nuls = torch.sum(text_idx[i, :] != null_idx)
        tokens[i, 0:non_nuls] = text_idx[i, 0:non_nuls]
        segments[i, 0:non_nuls] = segments_idx[0]
        tokens[i, non_nuls : non_nuls + cand_len] = cand_idx[i, :]
        segments[i, non_nuls : non_nuls + cand_len] = segments_idx[1]
    if use_cuda:
        tokens = tokens.cuda()
        segments = segments.cuda()
    return tokens, segments


def argsort(keys: List[Any], *lists: List[List[Any]], descending: bool = False):
    """
    Reorder each list in lists by the (descending) sorted order of keys.

    :param iter keys:
        Keys to order by.
    :param list[list] lists:
        Lists to reordered by keys's order.  Correctly handles lists and 1-D
        tensors.
    :param bool descending:
        Use descending order if true.

    :returns:
        The reordered items.
    """
    ind_sorted = sorted(range(len(keys)), key=lambda k: keys[k])
    if descending:
        ind_sorted = list(reversed(ind_sorted))
    output = []
    for lst in lists:
        # watch out in case we don't have torch installed
        if isinstance(lst, torch.Tensor):
            output.append(lst[ind_sorted])
        else:
            output.append([lst[i] for i in ind_sorted])
    return output


class IdentityLayer(torch.nn.Module):
    """
    Identity layer module.

    Useful for decoder-only Torch Generator agents.
    """

    def forward(self, xs):
        """
        Identity.
        """
        return xs


Chunk = TypeVar('Chunk')


PipelineWorkItem = namedtuple(
    'PipelineWorkItem', ['chunk_idx', 'layer_nos', 'next_device']
)


class PipelineHelper(object):
    """
    PipelineHelper assists with implementing pipelining in model parallelism.

    For a tutorial on model parallelism, as it's implemented in parts of ParlAI,
    see https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html.
    """

    @staticmethod
    def guess_split_size(item: Chunk, num_gpus: Optional[int] = None, dim=0) -> int:
        """
        Estimate the number of chunks we should split the batch into.

        Uses some silly heuristics.
        """
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()  # type: ignore
        if isinstance(item, torch.Tensor):
            return max(1, item.size(dim) // int(num_gpus * 2))
        elif isinstance(item, tuple):
            return PipelineHelper.guess_split_size(item[0], num_gpus)
        elif isinstance(item, dict):
            return PipelineHelper.guess_split_size(list(item.values())[0], num_gpus)
        raise TypeError(f'Cannot determine split size for {type(item)}')

    @staticmethod
    def split(item: Chunk, split_size: Optional[int] = None, dim=0) -> List[Chunk]:
        """
        Split a tensor or group of tensors into smaller chunks of the same type.

        :param item:
            The item being split. May be a Tensor, a tuple of Tensors, or a
            dictionary mapping str -> Tensor.
        :param split_size:
            The maximum size of each output chunk. If None, we will guess using
            heuristics
        :param dim:
            The dimension to split along.
        """
        if split_size is None:
            split_size = PipelineHelper.guess_split_size(item)

        if isinstance(item, torch.Tensor):
            # base case, just split the tensor
            return list(torch.split(item, split_size, dim))
        elif isinstance(item, tuple):
            # We start with Tuple[Tensor] and we return List[Tuple[Tensor]]
            return list(zip(*(PipelineHelper.split(i, split_size, dim) for i in item)))
        elif isinstance(item, dict):
            if item == {}:
                # terrible edge case: the empty dict. return an infinite list
                # of empty dicts and we'll figure out its correct size later
                return itertools.repeat({})
            # we start with Dict[key,tensor]
            # we map it to d: Dict[key, List[Tensor]], where we have split each mapping
            d = {k: PipelineHelper.split(v, split_size, dim) for k, v in item.items()}
            # now we transpose it and return List[Dict[key, Tensor]]
            return [
                dict(zip(d.keys(), values))  # type: ignore
                for values in zip(*(d[k] for k in d.keys()))
            ]
        else:
            raise TypeError(f"Cannot split type {type(item)}")

    @staticmethod
    def join(items: List[Chunk], dim=0) -> Chunk:
        """
        Join chunks back together, the inverse of split.

        :param items:
            All the output chunks. Each chunk may be a tensor or a group of
            tensors.
        :param dim:
            The dimension to join along.
        """
        if len(items) == 0:
            raise IndexError("Cannot rejoin an empty list of chunks.")
        item0 = items[0]
        if isinstance(item0, torch.Tensor):
            # base case
            return torch.cat(items, dim=dim)  # type: ignore
        elif isinstance(item0, tuple):
            return tuple(PipelineHelper.join(x, dim=dim) for x in zip(*items))  # type: ignore
        elif isinstance(item0, dict):
            keys = item0.keys()
            return {  # type: ignore
                k: PipelineHelper.join([c[k] for c in items], dim=dim)  # type: ignore
                for k in keys
            }
        else:
            raise TypeError(f'Cannot join list of type {type(item0)}')

    @staticmethod
    def layer_assignment(
        layer_no: int, num_layers: int, num_gpus: Optional[int] = None
    ) -> str:
        """
        Determine which device a layer should be on.

        :param layer_no:
            0-indexed layer number
        :param num_layers:
            Total number of layers to parallelize
        :param num_gpus:
            Number of gpus to distribute over. If None, use all available devices.
        :returns:
            A specific device, e.g. "cuda:0" or "cuda:3".
        """
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()  # type: ignore
        assert isinstance(num_gpus, int)
        device_no = layer_no // int(math.ceil(num_layers / num_gpus))
        return f'cuda:{device_no}'

    @staticmethod
    def make_parallel(layers: torch.nn.ModuleList) -> torch.nn.ModuleList:
        """
        Make a list of modules model parallel.
        """
        for layer_no, layer in enumerate(layers):
            layer_gpu = PipelineHelper.layer_assignment(layer_no, len(layers))
            layer._mp_gpu = layer_gpu  # type: ignore
            layers[layer_no] = layer.to(layer_gpu)
        return layers

    @staticmethod
    def chunk_to(chunk: Chunk, device: str) -> Chunk:
        """
        Move the chunk to the device.

        Handles chunks which are groups of tensors.
        """
        if isinstance(chunk, torch.Tensor):
            return chunk.to(device)  # type: ignore
        elif isinstance(chunk, tuple):
            return tuple(PipelineHelper.chunk_to(c, device) for c in chunk)  # type: ignore
        elif isinstance(chunk, dict):
            return {k: PipelineHelper.chunk_to(v, device) for k, v in chunk.items()}  # type: ignore
        else:
            raise TypeError('chunk_to only compatible with tensors, tuples or dicts.')

    @staticmethod
    def schedule_work_items(layers: torch.nn.ModuleList, chunks: List[Chunk]):
        """
        Iterate through chunks and layers that should be pipelined.

        Each iteration of this generator yields the following properties:

            - layer_nos: a list of indices of layers for you to forward through
            - chunk_idx: the index of the chunk we are manipulating. Use this
              if you need to update chunk representations.
            - next_device: where the chunk should be moved to AFTER the layer
              computation is done.
        """
        # We want to pipeline our computations so that each GPU is working on
        # chunks of the problem at the same of the time. The load of the will
        # look like this, assuming there are 5 chunks (A, B, C, D, E) and 4
        # GPUs. Each slot fill means that gpu is working on that chunk.
        #
        #         +-----------------+
        #         |       Time      |
        #         | 1 2 3 4 5 6 7 8 |
        # +-------+-----------------+
        # |  G  0 | A B C D E       |
        # |  P  1 |   A B C D E     |
        # |  U  2 |     A B C D E   |
        # |     3 |       A B C D E |
        # +-------+-----------------+
        #
        # Note that some GPUs will be idle much of the time. In reality, we
        # will use 1.5 * num_gpus as the number of chunks, to minimize idle
        # time.
        num_chunks = len(chunks)
        for l in layers:
            if not hasattr(l, '_mp_gpu'):
                raise RuntimeError(
                    'You must run PipelineHelper.make_parallel on the ModuleList '
                    'before you can use iterate_layers_chunks.'
                )

        # devices maps device_idx -> (device, [(i, layers[i], ...])
        # for example, if devices is 2 and there are 4 layers, we will have
        # devices = {
        #   0: ('cuda:0', [0, 1j]),
        #   1: ('cuda:1', [2, 3]]),
        # }
        devices = {
            device_idx: (dev, list(grp))
            for device_idx, (dev, grp) in enumerate(
                itertools.groupby(range(len(layers)), lambda x: layers[x]._mp_gpu)
            )
        }
        num_timesteps = len(devices) + num_chunks
        for timestep in range(num_timesteps):
            for chunk_idx in range(num_chunks):
                device_idx = timestep - chunk_idx
                if device_idx >= 0 and device_idx < len(devices):
                    dev, layers_nos = devices[device_idx]
                    next_device, _ = devices[(device_idx + 1) % len(devices)]
                    assert device_idx in devices
                    yield PipelineWorkItem(
                        chunk_idx=chunk_idx,
                        layer_nos=layers_nos,
                        next_device=next_device,
                    )
