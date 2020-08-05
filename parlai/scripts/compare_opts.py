#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json

from parlai.core.opt import Opt


def compare_opts(opt_path_1: str, opt_path_2: str, load_raw: bool = False) -> str:
    """
    Super simple script to compare the contents of two .opt files.

    Return the formatted text comparing the two .opts, for printing.
    """

    # Loading opt files
    if load_raw:
        with open(opt_path_1) as f1:
            opt1 = json.load(f1)
        with open(opt_path_2) as f2:
            opt2 = json.load(f2)
    else:
        opt1 = Opt.load(opt_path_1)
        opt2 = Opt.load(opt_path_2)

    outputs = list()

    outputs.append('\nArgs only found in opt 1:')
    opt1_only_keys = sorted([k for k in opt1.keys() if k not in opt2.keys()])
    for key in opt1_only_keys:
        outputs.append(f'{key}: {opt1[key]}')

    outputs.append('\nArgs only found in opt 2:')
    opt2_only_keys = sorted([k for k in opt2.keys() if k not in opt1.keys()])
    for key in opt2_only_keys:
        outputs.append(f'{key}: {opt2[key]}')

    outputs.append('\nArgs that are different in both opts:')
    keys_with_conflicting_values = sorted(
        [k for k, v in opt1.items() if k in opt2.keys() and v != opt2[k]]
    )
    for key in keys_with_conflicting_values:
        if isinstance(opt1[key], dict) and isinstance(opt2[key], dict):
            outputs.append(f'{key} (printing only non-matching values in each dict):')
            all_inner_keys = sorted(
                list(set(opt1[key].keys()).union(set(opt2[key].keys())))
            )
            for inner_key in all_inner_keys:
                if (
                    inner_key not in opt1[key]
                    or inner_key not in opt2[key]
                    or opt1[key][inner_key] != opt2[key][inner_key]
                ):
                    outputs.append(f'\t{inner_key}:')
                    outputs.append(
                        f'\t\tIn opt 1: {opt1[key].get(inner_key, "<MISSING>")}'
                    )
                    outputs.append(
                        f'\t\tIn opt 2: {opt2[key].get(inner_key, "<MISSING>")}'
                    )
        else:
            outputs.append(f'{key}:')
            outputs.append(f'\tIn opt 1: {opt1[key]}')
            outputs.append(f'\tIn opt 2: {opt2[key]}')

    return '\n'.join(outputs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('opt_path_1', type=str, help="Path to the first .opt file")
    parser.add_argument('opt_path_2', type=str, help="Path to the second .opt file")
    parser.add_argument(
        '-r',
        '--load-raw',
        action='store_true',
        help='Load using JSON instead of with Opt.load()',
    )
    args = parser.parse_args()

    output = compare_opts(
        opt_path_1=args.opt_path_1, opt_path_2=args.opt_path_2, load_raw=args.load_raw
    )
    print(output)
