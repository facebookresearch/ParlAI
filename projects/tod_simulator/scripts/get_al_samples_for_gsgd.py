#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Quick script for dumping out relevant conversation ids from GoogleSGD.
"""

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.tasks.google_sgd_simulation_splits.agents import GoogleSgdOutDomainParser
from parlai.core.tod.tod_agents import TodStructuredDataParser

import parlai
import os
import json
import random

PARLAI_DATA_PATH = os.path.dirname(os.path.dirname(parlai.__file__)) + "/data"


class GrabEpisodes(GoogleSgdOutDomainParser, TodStructuredDataParser):
    def get_agent_type_suffix(self):
        return "GrabEpisodes"


def setup_args(parser=None):
    if not parser:
        parser = ParlaiParser(False, False)
    group = parser.add_argument_group("Get active learning samples script")
    group.add_argument(
        "--find-random-al-samples",
        type=bool,
        default=False,
        help="Get active learning samples randomly or few shot",
    )
    group.add_argument(
        "--input-processed-stats",
        type=str,
        help="Processed stats file from the `get_passing_only` script",
    )
    group.add_argument(
        "--processed-stats-section",
        type=str,
        default="MISSES",
        help="Which section from the `get_passing_only` script will we use to rank. (MISSES and FRACTIONAL current options)",
    )
    group.add_argument(
        "--num-apis-to-get",
        default=8,
        type=int,
        help="Number of api descriptions we want to find",
    )
    group.add_argument(
        "--existing-al-files",
        nargs="*",
        type=str,
        help="Existing active learning files (ie, for running multiple iterations of learning)",
    )
    group.add_argument(
        "--cumulative-al",
        type=bool,
        default=True,
        help="Uses active learning files from '--existing-al-files' cumulatively (as in, will append all prior dialog ids for next, rather than excluding)",
    )
    group.add_argument(
        "--al-output-file",
        default=None,
        help="Output file. Will put into 'results' in active run directory otherwise.",
    )
    return parser


@register_script("get_al_samples_for_gsgd_script")
class GetAlSamplesForGsgdScript(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        existing_al_ids = self.get_existing_al_ids()
        # NOTE: The inidivdual get_al_samples funcitons are responsible for dealing with cumulative
        if self.opt["find_random_al_samples"]:
            save_me = self.get_al_samples_random(existing_al_ids)
        else:
            save_me = self.get_al_samples_from_processed(existing_al_ids)
        out_file = self.opt.get("al_output_file")
        if not out_file:
            out_file = "result"
        with open(out_file, "w+") as f:
            json.dump(save_me, f, indent=4)
        print("Saved AL samples to ", out_file)

    def get_al_samples_random(self, existing_al_ids):
        save_me = {}
        for datatype in ["train", "valid", "test"]:
            unfiltered_episodes = self.get_gsgd_episodes_for_datatype(datatype)
            filtered_episodes = [
                x
                for x in unfiltered_episodes
                if x.extras["dialogue_id"] not in existing_al_ids
            ]
            samples = random.Random(42).sample(
                filtered_episodes, self.opt["num_apis_to_get"]
            )
            if self.opt["cumulative_al"]:
                old = [
                    x
                    for x in unfiltered_episodes
                    if x.extras["dialogue_id"] in existing_al_ids
                ]
                samples.extend(old)
            save_me[datatype] = {
                episode.extras["dialogue_id"]: episode.goal_calls_utt
                for episode in samples
            }
        return save_me

    def get_al_samples_from_processed(self, existing_al_ids):
        wanted_apis = self.get_wanted_apis()
        save_me = {}
        for datatype in ["train", "valid", "test"]:
            found = [False] * len(wanted_apis)
            save_me[datatype] = {}
            for episode in self.get_gsgd_episodes_for_datatype(datatype):
                if episode.extras["dialogue_id"] in existing_al_ids.get(datatype, []):
                    if self.opt["cumulative_al"]:
                        save_me[datatype][
                            episode.extras["dialogue_id"]
                        ] = episode.goal_calls_utt
                    continue
                here = False
                for idx, val in enumerate(found):
                    if not val and not here:
                        here = True
                        for api_name_slots in wanted_apis[idx]:
                            api_name = api_name_slots[0]
                            slots = api_name_slots[1]
                            if api_name in episode.goal_calls_utt:
                                for slot in slots:
                                    if slot not in episode.goal_calls_utt:
                                        here = False
                            else:
                                here = False
                        if here:
                            found[idx] = True
                            save_me[datatype][
                                episode.extras["dialogue_id"]
                            ] = episode.goal_calls_utt
        return save_me

    def get_gsgd_episodes_for_datatype(self, datatype):
        datapath = PARLAI_DATA_PATH
        if "datapath" in self.opt:
            datapath = self.opt["datapath"]
        elif "parlai_home" in self.opt:
            datapath = self.opt["parlai_home"] + "/data"
        opt = {
            "datatype": datatype,
            "datapath": datapath,
            "gsgd_domains": "all",
            "n_shot": -1,
            "episodes_randomization_seed": -1,
        }
        return GrabEpisodes(opt).episodes

    def get_existing_al_ids(self):
        existing_al_files = self.opt.get("existing_al_files")
        if existing_al_files is None:
            return {}
        existing_al_ids = {}
        for existing in existing_al_files:
            with open(existing) as f:
                data = json.load(f)
            for datatype, blob in data.items():
                if datatype not in existing_al_ids:
                    existing_al_ids[datatype] = []
                for dialog_id in blob:
                    existing_al_ids[datatype].append(dialog_id)
        return existing_al_ids

    def get_wanted_apis(self):
        unprocessed_lines = []
        with open(self.opt["input_processed_stats"]) as f:
            lines = f.readlines()
            idx = 0
            while self.opt["processed_stats_section"] not in lines[idx]:
                idx += 1
            unprocessed_lines = lines[idx + 1 : idx + 1 + self.opt["num_apis_to_get"]]

        processed = []
        for row_raw in unprocessed_lines:
            print(row_raw)
            row = json.loads(json.loads(row_raw.strip())[0])
            slots = [x for x in row if "api_name" not in x]
            for x in row:
                if "api_name" in x:
                    processed.append((x.replace("api_name:", ""), slots))
        return processed


if __name__ == "__main__":
    GetAlSamplesForGsgdScript.main()
