#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base script for running TOD model-model chats.
"""

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script

from projects.tod_simulator.scripts.get_passing_only import GetPassingOnlyScript

import glob


@register_script("do_get_passing_only_on_dir")
class DoGetPassingOnlyOnDirScript(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(False, False)
        parser.add_argument("-p", "--path", required=True, type=str)
        parser.add_argument(
            "--filter-call-attempts",
            default=True,
            help="when True, only counts as 'passing' if System made exactly same # of api calls as goals",
        )
        return parser

    def run(self):
        opt = self.opt
        path = opt["path"]

        # assumes standard naming from the `model_consts` set of scripts
        base_paths = [
            x.replace("_conversations.jsonl", "")
            for x in glob.glob(f"{path}/*_conversations.jsonl")
        ]

        for to_run in base_paths:
            convo_path = to_run + "_conversations.jsonl"
            report_path = to_run + ".json"

            here_opt = {
                "convo_path": convo_path,
                "report_path": report_path,
                "print_to_file": True,
                "filter_call_attempts": opt["filter_call_attempts"],
            }
            GetPassingOnlyScript._run_kwargs(here_opt)
            print("done with ", convo_path)


if __name__ == "__main__":
    DoGetPassingOnlyOnDirScript.main()
