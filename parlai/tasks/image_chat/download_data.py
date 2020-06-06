#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from parlai.tasks.image_chat.build import build


def parse_args():
    """
    Wrapper to parse CLI arguments.

    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dp", "--datapath", default="/tmp", help="Path where to save data."
    )

    args = parser.parse_args()
    # opts is dic in parlai
    args = vars(args)

    return args


if __name__ == "__main__":
    opt = parse_args()
    # Only datapath is required by build.
    # Using build function to check the version and
    # internal hash
    build(opt)
