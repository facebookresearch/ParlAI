# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch.autograd import Variable

from itertools import zip_longest

def get_collate_fn(cuda=False):
    def _collate_fn(data):
        batch = dict()
        for k in data[0].keys():
            #for each k key at ith row
            k_data = [data[i][k] for i in range(len(data))]

            if k in ['textrecog', 'landmarks']:
                batch[k], _ = list_to_tensor(k_data)
            if k in ['goldstandard', 'actions']:
                batch[k], batch[k+'_mask'] = list_to_tensor(k_data)
            if k  == 'utterance':
                batch['utterance'], batch['utterance_mask'] = list_to_tensor(k_data)
            if k in ['target']:
                batch[k] = torch.LongTensor(k_data)
            if k in ['resnet', 'weight']:
                batch[k] = torch.FloatTensor(k_data)
            if k == 'fasttext':
                batch[k], _ = list_to_tensor(k_data, tensor_type=torch.FloatTensor)

        batch['episode_done'] = list_to_tensor([1 for i in range(len(data))])

        return to_variable(batch, cuda=False)
    return _collate_fn

def get_max_dimensions(arr):
    """Recursive function to calculate max dimensions of
       tensor (given a multi-dimensional list of arbitrary depth)
    """
    if not isinstance(arr, list):
        return []

    if len(arr) == 0:
        return [0]

    dims = None
    for a in arr:
        if dims is None:
            dims = get_max_dimensions(a)
        else:
            dims = [max(x, y) for x, y in zip_longest(dims, get_max_dimensions(a), fillvalue=0)]
    dims = [len(arr)] + dims
    return dims if 0 not in dims[1:] else []


def fill(ind, data_arr, value_tensor, mask_tensor):
    """Recursively fill tensor with values from multidimensional array
    """
    if not isinstance(data_arr, list):
        value_tensor[tuple(ind)] = data_arr
        mask_tensor[tuple(ind)] = 1.0
    else:
        for i, a in enumerate(data_arr):
            fill(ind + [i], a, value_tensor, mask_tensor)

def list_to_tensor(arr, pad_value=0, tensor_type=torch.LongTensor):
    """Convert multi-dimensional array into tensor. Also returns mask.
    """
    dims = get_max_dimensions(arr)

    val_tensor = tensor_type(size=dims).fill_(pad_value)
    mask_tensor = torch.FloatTensor(size=dims).zero_()
    fill([], arr, val_tensor, mask_tensor)
    return val_tensor, mask_tensor

def to_variable(obj, cuda=True):
    if torch.is_tensor(obj):
        var = Variable(obj)
        if cuda:
            var = var.cuda()
        return var
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [to_variable(x, cuda=cuda) for x in obj]
    if isinstance(obj, dict):
        return {k: to_variable(v, cuda=cuda) for k, v in obj.items()}
