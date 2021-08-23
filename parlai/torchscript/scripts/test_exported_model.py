#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import List

import torch.jit

from parlai.utils.io import PathManager


def test_exported_model(scripted_model_file: str, inputs: List[str]):

    with PathManager.open(scripted_model_file, "rb") as f:
        scripted_module = torch.jit.load(f)

    print('\nGenerating given the scripted module:')
    context = []
    for input_ in inputs:
        print(' TEXT: ' + input_)
        context.append(input_)
        label = scripted_module('\n'.join(context))
        print("LABEL: " + label)
        context.append(label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-smf',
        '--scripted-model-file',
        type=str,
        help='Where to load the scripted model checkpoint from',
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="hello world",
        help="Test input string to pass into the encoder of the scripted model. Separate lines with a pipe",
    )
    args = parser.parse_args()
    test_exported_model(
        scripted_model_file=args.scripted_model_file, inputs=args.input.split('|')
    )
