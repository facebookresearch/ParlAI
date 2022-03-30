#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base script for running TOD model-model chats.
"""
from parlai.core.metrics import InterDistinctMetric

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script

import glob
import json


@register_script("get_interdistinct_on_conversations")
class GetInterdistinctOnConversationsScript(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(False, False)
        parser.add_argument("-p", "--path", required=True, type=str)
        return parser

    def run(self):
        opt = self.opt
        paths = glob.glob(f"{opt['path']}/*_conversations.jsonl")

        for path in paths:
            name = path.split("/")[-1]
            sys_utt_one = InterDistinctMetric.compute("", 1)
            sys_utt_two = InterDistinctMetric.compute("", 2)
            user_utt_one = InterDistinctMetric.compute("", 1)
            user_utt_two = InterDistinctMetric.compute("", 2)

            with open(path) as f:
                for line_raw in f:
                    line = json.loads(line_raw)["dialog"]
                    for turn in line:
                        if len(turn) < 4:
                            continue
                        user_utt = turn[0]["text"]
                        if user_utt.startswith("APIS: "):
                            continue
                        user_utt_one += InterDistinctMetric.compute(user_utt, 1)
                        user_utt_two += InterDistinctMetric.compute(user_utt, 2)

                        sys_utt = turn[3]["text"]
                        sys_utt_one += InterDistinctMetric.compute(sys_utt, 1)
                        sys_utt_two += InterDistinctMetric.compute(sys_utt, 2)

            print(
                ",".join(
                    [
                        str(x)
                        for x in [
                            name,
                            user_utt_one.value(),
                            user_utt_two.value(),
                            sys_utt_one.value(),
                            sys_utt_two.value(),
                        ]
                    ]
                )
            )


if __name__ == "__main__":
    GetInterdistinctOnConversationsScript.main()
