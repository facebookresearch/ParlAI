#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import json
import os

def make_model_args(
    save_root, system_tuple, user_tuple, datatype, extra_report_filename_suffixes=""
):
    return (
        f" --system-model-file {system_tuple[1]} {system_tuple[2]}"
        f" --user-model-file {user_tuple[1]} {user_tuple[2]}"
        f" --datatype {datatype}"
        f" --report-filename {save_root}/{system_tuple[0]}_-_{user_tuple[0]}{extra_report_filename_suffixes}"
        f" --world-logs {save_root}/{system_tuple[0]}_-_{user_tuple[0]}{extra_report_filename_suffixes}"
    )


def make_path(folder, model_key):
    path = os.path.join(BASE_DIR, folder, model_key, "model")
    if not os.path.isfile(path):
        raise RuntimeError(f"Path {path} does not exists as a model!")
    return path

def get_level_from_sweep_dir(path):
    path = path.replace("_", "")
    num_idx = path.find("level")
    if num_idx == -1:
        raise RuntimeError(
            "Did not find 'level' in the path for determining what level of RL we are in. Path: "
            + path
        )
    num_idx += len("level")
    if not path[num_idx].isdigit():
        raise RuntimeError(
            "Could not find a digit to represent the level in the path. Path: " + path
        )
    if path[num_idx + 1].isdigit():
        raise RuntimeError(
            "This code for finding the level is dumb and only works for one-digit integers. Path: "
            + path
        )
    return int(str(path[num_idx]))
