#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import json
from parlai.core.teachers import DialogTeacher
from .build import build


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(
        opt['datapath'], 'reframe_thoughts', 'reframe_thoughts_dataset', dt + '.txt'
    )


class ReframeThoughtsTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = "reframe_thoughts"
        opt['datafile'] = _path(opt)
        super().__init__(opt, shared)

    def setup_data(self, datafile):
        all_data = [json.loads(line.strip()) for line in open(datafile)]
        for data in all_data:
            persona = data["persona"]
            pattern = data["pattern"] + " (" + data["pattern_def"] + ")"
            thought = data["thought"]
            persona_token = ""
            pattern_token = "[PAT]"
            thought_token = "[THT]"
            for reframe in data["reframes"]:
                input = " ".join(
                    [
                        persona_token,
                        persona,
                        pattern_token,
                        pattern,
                        thought_token,
                        thought,
                    ]
                ).strip()
                yield (input, reframe['reframe']), True


class DefaultTeacher(ReframeThoughtsTeacher):
    pass
