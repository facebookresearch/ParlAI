#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn

from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import TorchAgent


def setup_args() -> ParlaiParser:
    parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
    parser.add_argument(
        '-smf',
        '--scripted-model-file',
        type=str,
        default='_scripted.pt',
        help='Where the scripted model checkpoint will be saved',
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="hello world",
        help="Test input string to pass into the encoder of the scripted model. Separate lines with a pipe",
    )
    return parser
