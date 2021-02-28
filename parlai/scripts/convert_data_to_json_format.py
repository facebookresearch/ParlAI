#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Converts data used in a task to json format. (Same as "Conversation" class; ie, for use
in ACUTE-eval)

Specify the task with `-t`. By default, this code will save to a file with prefix "tmp".
To change the prefix, set `--world-logs`.
"""

from parlai.core.script import register_script
from parlai.scripts.eval_model import EvalModel


@register_script('convert_to_json', hidden=True)
class DumpDataToConversations(EvalModel):
    @classmethod
    def setup_args(cls):
        parser = EvalModel.setup_args()
        parser.description = 'Convert data to json format'
        parser.set_defaults(model="repeat_label")
        parser.set_defaults(world_logs="tmp")
        return parser


if __name__ == '__main__':
    DumpDataToConversations.main()
