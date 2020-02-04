#!/usr/bin/env python3

from parlai.tasks.image_chat.build import build
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dp", "--datapath", default="/tmp",
        help="Path where to save data."
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
