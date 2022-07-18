#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base script for running TOD model-model chats.
"""

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
import glob
import json
import os


class GetQuickEvalStatsDirScript(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(False, False)
        parser.add_argument("-p", "--path", required=True, type=str)
        return parser

    def run(self):
        opt = self.opt
        path = opt["path"]

        stuff = {}

        for eval_stats in glob.glob(f"{path}/*/eval_stats.json"):
            with open(eval_stats) as f:
                eval_data = json.load(f)

            base_path = os.path.abspath(eval_stats).replace("eval_stats.json", "/")

            base = ""
            multitask = ""
            if not os.path.isfile(base_path + "run.sh"):
                continue
            with open(base_path + "run.sh") as run_file:
                for line in run_file:
                    if "zoo:bart" in line:
                        base = "BartOnly"

            with open(base_path + "model.opt") as opt_file:
                print(base_path)
                opt = json.load(opt_file)
                lr = opt["learningrate"]
                multitask = opt.get("multitask_weights", "")
                yes_api = opt.get("api_schemas", False)
            if multitask != "":
                multitask = "_" + "".join([str(x) for x in multitask])

            orig_path = "/".join(os.path.abspath(base_path).split("/")[-3:])

            ppl = eval_data["report"].get("ppl", "")
            token_em = eval_data["report"].get("token_em", "")
            jga = eval_data["report"].get("jga", "")
            jga_n = eval_data["report"].get("jga_noempty", "")
            jga_e = eval_data["report"].get("jga_empty", "")

            if yes_api:  # hack cause I'm annoyed at things being in wrong alpha order
                yes_api = "True"
            else:
                yes_api = "false"

            maybe_nshot = ""
            root = base_path
            eval_file = glob.glob(f"{root}/*mm_eval*.json")
            if len(eval_file) > 0 and (
                "percent" in eval_file[0] or "nshot" in eval_file[0]
            ):
                maybe_nshot = eval_file[0][len(root + "mm_eval_") : -len(".json")]

            base += f"{len(opt.get('task').split(','))}-{yes_api}-APIS__{maybe_nshot}{base}_{lr}{multitask}"
            metrics = [orig_path, base, ppl, token_em, jga, jga_n, jga_e]

            if os.path.isfile(eval_stats.replace("eval_stats", "user_eval_stats")):
                with open(eval_stats.replace("eval_stats", "user_eval_stats")) as f:
                    metrics.append(json.load(f)["report"]["ppl"])
            else:
                metrics.append("DNE")

            metrics.append(" ")

            root = base_path
            maybe_mm_stats = glob.glob(f"{root}/mm_eval")
            if len(maybe_mm_stats) == 0:
                maybe_mm_stats = glob.glob(f"{root}/*mm_eval*.json")
            if len(maybe_mm_stats) == 0:
                maybe_mm_stats = glob.glob(f"{root}/*Mult*.json")
            if len(maybe_mm_stats) > 0:
                with open(maybe_mm_stats[0]) as f:
                    report = json.load(f)["report"]
                    if "synthetic_task_success" in report:
                        tsr = report["synthetic_task_success"]
                    elif "all_goals_hit" in report:
                        tsr = report["all_goals_hit"]
                    else:
                        tsr = "DNE"
                    metrics.append(tsr)
            else:
                metrics.append("DNE")

            metrics.append(" ")

            if os.path.isfile(eval_stats.replace("eval_stats", "in_eval_stats")):
                with open(eval_stats.replace("eval_stats", "in_eval_stats")) as f:
                    blob = json.load(f)["report"]
                    metrics.append(blob.get("ppl", ""))
                    metrics.append(blob.get("jga", ""))
                    metrics.append(blob.get("jga_noempty", ""))
                    metrics.append(blob.get("jga_empty", ""))
            else:
                metrics.append("DNE")
                metrics.append("DNE")
                metrics.append("DNE")
                metrics.append("DNE")

            if os.path.isfile(eval_stats.replace("eval_stats", "in_user_eval_stats")):
                with open(eval_stats.replace("eval_stats", "in_user_eval_stats")) as f:
                    metrics.append(json.load(f)["report"]["ppl"])
            else:
                metrics.append("DNE")

            if os.path.isfile(eval_stats.replace("eval_stats", "test_eval_stats")):
                with open(eval_stats.replace("eval_stats", "test_eval_stats")) as f:
                    blob = json.load(f)["report"]
                    metrics.append(blob.get("ppl", ""))
                    metrics.append(blob.get("jga", ""))
                    metrics.append(blob.get("jga_noempty", ""))
                    metrics.append(blob.get("jga_empty", ""))
            else:
                metrics.append("DNE")
                metrics.append("DNE")
                metrics.append("DNE")
                metrics.append("DNE")

            if os.path.isfile(eval_stats.replace("eval_stats", "test_user_eval_stats")):
                with open(
                    eval_stats.replace("eval_stats", "test_user_eval_stats")
                ) as f:
                    metrics.append(json.load(f)["report"]["ppl"])
            else:
                metrics.append("DNE")

            stuff[base] = ",".join([str(x) for x in metrics])

        ordering = [
            "orig_path",
            "base",
            "sys_ppl",
            "token_em",
            "jga",
            "jga_noempty",
            "jga_empty",
            "user_ppl",
            "space",
            "sTsr",
            "space",
            "in_sys_ppl",
            "in_jga",
            "in_jga_noempty",
            "in_jga_empty",
            "in_user_ppl",
            "test_sys_ppl",
            "test_jga",
            "test_jga_noempty",
            "test_jga_empty",
            "test_user_ppl",
        ]
        print(",".join(ordering))
        for key in sorted(stuff.keys()):
            print(stuff[key])


if __name__ == "__main__":
    GetQuickEvalStatsDirScript.main()
