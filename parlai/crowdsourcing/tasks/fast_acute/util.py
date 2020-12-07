#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from typing import Iterable, List, Tuple, Union


def get_hashed_combo_path(
    root_dir: str,
    subdir: str,
    task: str,
    combos: Iterable[Union[List[str], Tuple[str, str]]],
) -> str:
    """
    Return a unique path for the given combinations of models.

    :param root_dir: root save directory
    :param subdir: immediate subdirectory of root_dir
    :param task: the ParlAI task being considered
    :param combos: the combinations of models being compared
    """

    # Sort the names in each combo, as well as the overall combos
    sorted_combos = []
    for combo in combos:
        assert len(combo) == 2
        sorted_combos.append(tuple(sorted(combo)))
    sorted_combos = sorted(sorted_combos)

    os.makedirs(os.path.join(root_dir, subdir), exist_ok=True)
    path = os.path.join(
        root_dir,
        subdir,
        hashlib.sha1(
            '___and___'.join(
                [f"{m1}vs{m2}.{task.replace(':', '_')}" for m1, m2 in sorted_combos]
            ).encode('utf-8')
        ).hexdigest()[:10],
    )
    return path
