#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from parlai.utils import pickle
from parlai.utils.io import PathManager
from parlai.utils.torch import atomic_save


def remove_projection_matrices(model_file: str):
    """
    Remove all projection matrices used for distillation from the model and re-save it.
    """

    print(f'Creating a backup copy of the original model at {model_file}._orig.')
    PathManager.copy(model_file, f'{model_file}._orig')

    print(f"Loading {model_file}.")
    with PathManager.open(model_file, 'rb') as f:
        states = torch.load(f, map_location=lambda cpu, _: cpu, pickle_module=pickle)

    print('Deleting projection matrices.')
    orig_num_keys = len(states['model'])
    states['model'] = {
        key: val
        for key, val in states['model'].items()
        if key.split('.')[0]
        not in ['encoder_proj_layer', 'embedding_proj_layers', 'hidden_proj_layers']
    }
    new_num_keys = len(states['model'])
    print(f'{orig_num_keys-new_num_keys:d} model keys removed.')

    print(f"Saving to {model_file}.")
    atomic_save(states, model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_file', type=str, help='Path to model to remove projection matrices from'
    )
    args = parser.parse_args()
    remove_projection_matrices(args.model_file)
