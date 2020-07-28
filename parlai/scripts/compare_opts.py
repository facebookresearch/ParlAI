#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json


def compare_opts(opt_path_1: str, opt_path_2: str):
    """
    Super simple script to compare the contents of two .opt files.
    """

    # Loading opt files
    with open(opt_path_1) as f1:
        opt1 = json.load(f1)
    with open(opt_path_2) as f2:
        opt2 = json.load(f2)

    print('\nArgs only found in opt 1:')
    opt1_only_keys = sorted([k for k in opt1.keys() if k not in opt2.keys()])
    for key in opt1_only_keys:
        print(f'\t{key}: {opt1[key]}')

    print('\nArgs only found in opt 2:')
    opt2_only_keys = sorted([k for k in opt2.keys() if k not in opt1.keys()])
    for key in opt2_only_keys:
        print(f'\t{key}: {opt2[key]}')

    print('\nArgs that are different in both opts:')
    keys_with_conflicting_values = sorted(
        [k for k, v in opt1.items() if k in opt2.keys() and v != opt2[k]]
    )
    for key in keys_with_conflicting_values:
        print(f'\t{key}:')
        print(f'\t\tIn opt 1: {opt1[key]}')
        print(f'\t\tIn opt 2: {opt2[key]}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('opt_path_1', type=str, help="Path to the first .opt file")
    parser.add_argument('opt_path_2', type=str, help="Path to the second .opt file")
    args = parser.parse_args()

    compare_opts(opt_path_1=args.opt_path_1, opt_path_2=args.opt_path_2)
