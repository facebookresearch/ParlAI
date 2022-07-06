#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base script for running TOD model-model chats.
"""

from collections import defaultdict
from copy import deepcopy
import json
import sys
from shutil import copyfile

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.tod.tod_core import STANDARD_DONE
from parlai.utils.io import PathManager

from parlai.core.tod.tod_core import SerializationHelpers


@register_script("get_passing_only")
class GetPassingOnlyScript(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(False, False)
        parser.add_argument("--cut-first-400", default=False)
        parser.add_argument("--convo-path", required=True)
        parser.add_argument(
            "--report-path",
            required=True,
            help="path of the report saved from the tod_metrics_script",
        )
        parser.add_argument(
            "--print-to-file",
            default=False,
            help="save the results to a file (by hackishly redirecting stdout)",
        )
        parser.add_argument(
            "--filter-call-attempts",
            default=True,
            help="when True, only counts as 'passing' if System made exactly the same # of api calls as # of goals",
        )
        return parser

    def run(self):
        opt = self.opt

        with PathManager.open(opt["report_path"]) as r:
            print(opt["report_path"])
            report = json.load(r)["report"]
        tod_metrics = report["tod_metrics"]

        if opt["cut_first_400"]:
            tod_metrics = tod_metrics[400:]

        infile_base = opt["convo_path"].replace(".jsonl", "")
        outfile_base = infile_base + "_processed"

        copyfile(f"{infile_base}.metadata", f"{outfile_base}.metadata")

        if opt["print_to_file"]:
            print_file_path = infile_base.replace("_conversations", "_processed_stats")
            print_file = open(print_file_path, "w+")
            orig_stdout = sys.stdout
            sys.stdout = print_file

        found = defaultdict(lambda: 0)
        not_found = defaultdict(lambda: 0)

        with PathManager.open(f"{outfile_base}.jsonl", "w") as out:
            with PathManager.open(opt["convo_path"]) as c:
                lines = c.readlines()
                if opt["cut_first_400"]:
                    lines = lines[400:]
                print("len lines: ", len(lines))
                print("len tod_metrics: ", len(tod_metrics))
                for i, l in enumerate(lines):
                    if i >= len(tod_metrics):
                        break
                    goals = SerializationHelpers.str_to_goals(
                        tod_metrics[i]["goal"]["text"][len(STANDARD_DONE) :]
                    )
                    if len(goals) == 0:
                        print(goals)
                        continue
                    if tod_metrics[i]["synthetic_task_success"] == 1.0 and (
                        not self.opt["filter_call_attempts"]
                        or tod_metrics[i].get("api_call_attempts", 1)  # legacy
                        == len(goals)
                    ):
                        print(goals, tod_metrics[i]["api_call_attempts"], l)
                        self.api_string_add(found, goals, 1)
                        self.api_string_add(not_found, goals, 0)
                        out.write(l)
                    else:
                        self.api_string_add(not_found, goals, 1)
                        self.api_string_add(found, goals, 0)
            print("count found", sum(found.values()))
            print("count notfound", sum(not_found.values()))
            print(
                "============================ FOUND ======================\n",
                [
                    (k, v)
                    for k, v in sorted(found.items(), key=lambda x: x[1], reverse=True)
                ],
            )
            print(
                "============================= NOT FOUND ====================\n",
                [
                    (k, v)
                    for k, v in sorted(
                        not_found.items(), key=lambda x: x[1], reverse=True
                    )
                ],
            )

        biggest_misses = deepcopy(not_found)
        for f in found:
            biggest_misses[f] -= found[f]
        print(
            "======================= BIGGEST MISSES (# not found - # found) ======================\n",
            "\n".join(
                [
                    json.dumps((k, v))
                    for k, v in sorted(
                        biggest_misses.items(), key=lambda x: x[1], reverse=True
                    )
                ]
            ),
        )
        fraction = {}
        for k in not_found:
            total = not_found[k] + found[k]
            fraction[k] = (float(not_found[k]) / total, total)
        print(
            "=========================== BIGGEST FRACTIONAL DELTAS (not_found / total # )  ====================\n",
            "\n".join(
                [
                    json.dumps((k, v))
                    for k, v in sorted(
                        fraction.items(), key=lambda x: x[1], reverse=True
                    )
                ]
            ),
        )

        if opt["print_to_file"]:
            print_file.close()
            sys.stdout = orig_stdout

    def api_string_add(self, found_list, api_strings, val):
        for api_string in api_strings:
            if "api_name" not in api_string:
                continue
            api_name = api_string["api_name"]
            api_string["api_name:" + api_name] = ""
            api_name = api_string.pop("api_name")
            found_list[json.dumps(sorted(list(api_string)))] += val
            api_string["api_name"] = api_name


if __name__ == "__main__":
    GetPassingOnlyScript.main()
