#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Alias script around `parlai eval_model -m repeat_label`  to dump data to conversations format
"""

from parlai.core.script import ParlaiScript, register_script
from parlai.scripts.eval_model import EvalModel


@register_script('dump_data_to_conversations')
class DumpDataToConversations(EvalModel):
    @classmethod
    def setup_args(cls):
        parser = EvalModel.setup_args()
        parser.set_defaults(model="repeat_label")
        return parser


if __name__ == '__main__':
    DumpDataToConversations.main()
