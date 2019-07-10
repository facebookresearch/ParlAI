#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Interact with a pre-trained retrieval model.
"""
from parlai.scripts.interactive import setup_args, interactive


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='projects:wizard_of_wikipedia:interactive_retrieval',
        task='wizard_of_wikipedia',
    )
    opt = parser.parse_args(print_args=False)
    interactive(opt, print_parser=parser)
