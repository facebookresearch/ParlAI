#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility methods for dealing with torch code.
"""

from typing import Union, Optional, Tuple, Any, List, Sized


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
