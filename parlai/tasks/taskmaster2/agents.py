#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Taskmaster-2 implementation for ParlAI.

No official train/valid/test splits are available as of 2020-05-18, so we make our own
splits.
"""

from parlai.core.params import ParlaiParser
import os
import pandas as pd
from collections import Counter
from parlai.core.opt import Opt
import parlai.core.tod.tod_core as tod
from parlai.utils.misc import warn_once
import json
from typing import Optional
from parlai.utils.data import DatatypeHelper
from parlai.utils.io import PathManager

import parlai.tasks.taskmaster2.build as build_
import parlai.core.tod.tod_agents as tod_agents


DOMAINS = [
    "flights",
    "food-ordering",
    "hotels",
    "movies",
    "restaurant-search",
    "sports",
    "music",
]


class Taskmaster2Parser(tod_agents.TodStructuredDataParser):
    """
    Abstract data loader.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.add_argument(
            "--taskmaster2-domains",
            nargs="+",
            default=DOMAINS,
            choices=DOMAINS,
            help="Uses last passed in configuration.",
        )
        parser.add_argument(
            "--use-cumulative-api-calls",
            type=bool,
            default=True,
            help="Have API Call/API response turns only when an API response"
            "slot exist. Accumulate all API call slots with same API call name",
        )
        return super().add_cmdline_args(parser, partial_opt)

    def __init__(self, opt: Opt, shared=None):
        self.fold = DatatypeHelper.fold(opt["datatype"])
        opt["datafile"] = self.fold
        self.dpath = os.path.join(opt["datapath"], "taskmaster-2")
        if shared is None:
            warn_once(
                "Taskmaster2 is a beta dataset, and format may significantly change."
            )
            build_.build(opt)
        super().__init__(opt, shared)

    def _load_data(self, fold, domains):
        # load up the ontology
        ontologies = {}
        for section in domains:
            fn = os.path.join(self.dpath, section + ".onto.json")
            with PathManager.open(fn, "r") as f:
                ontologies.update(json.load(f))

        chunks = []
        for section in domains:
            with PathManager.open(os.path.join(self.dpath, section + ".json")) as f:
                subset = pd.read_json(f)
            subset["domain"] = section
            chunks.append(subset)
        chunks = pd.concat(chunks, axis=0)
        # deterministic shuffle data for splits
        chunks = chunks.sample(frac=1.0, random_state=42)
        split_size = len(chunks) // 10
        if fold == "train":
            chunks = chunks[: split_size * 8]
        elif fold == "valid":
            chunks = chunks[split_size * 8 : split_size * 9]
        elif fold == "test":
            chunks = chunks[split_size * 9 :]
        return chunks, ontologies

    def _parse_segment_to_slots(self, segment_list):
        result = {}
        for segment in segment_list:
            slot_name = segment["annotations"][0]["name"]
            slot_value = segment["text"]
            prefix_split_idx = slot_name.find(".")
            api_name = slot_name[:prefix_split_idx]
            slot_name = slot_name[prefix_split_idx + 1 :]
            result[slot_name] = slot_value
            result[tod.STANDARD_API_NAME_SLOT] = api_name
        return result

    def _get_utterance_and_api_call_for_speaker(self, speaker, utterances, idx):
        utts = []
        slots = {}
        while idx < len(utterances):
            here = utterances[idx]
            if here["speaker"] != speaker:
                break
            utts.append(here["text"])
            slots.update(self._parse_segment_to_slots(here.get("segments", [])))
            idx += 1
        return idx, "\n".join(utts), slots

    def _get_onto_list(self, onto_map, domain):
        results = []
        domain = domain.replace(
            "-", "_"
        )  # cause they changed it for restaurant-search >.>
        for data in onto_map[domain]:
            call = {}
            call[tod.STANDARD_API_NAME_SLOT] = data["prefix"]
            call[tod.STANDARD_OPTIONAL_KEY] = data[
                "annotations"
            ]  # make all args optional since not specified
            results.append(call)
        return results

    def setup_episodes(self, fold):
        """
        Parses into TodStructuredEpisode.
        """
        domains = self.opt.get("taskmaster2_domains", DOMAINS)
        chunks, ontologies = self._load_data(fold, domains)
        domains_cnt = Counter()
        episodes = []
        for _, row in chunks.iterrows():
            domains_cnt[row["domain"]] += 1
            utterances = row["utterances"][:]

            idx = 0
            rounds = []
            goal_calls = []
            if len(utterances) > 0 and utterances[0]["speaker"] == "ASSISTANT":
                idx, sys_utt, api_resp = self._get_utterance_and_api_call_for_speaker(
                    "ASSISTANT", utterances, idx
                )
                r = tod.TodStructuredRound(api_resp_machine=api_resp, sys_utt=sys_utt)
                rounds.append(r)

            cum_api_call = {}
            while idx < len(utterances):
                idx, user_utt, api_call = self._get_utterance_and_api_call_for_speaker(
                    "USER", utterances, idx
                )
                idx, sys_utt, api_resp = self._get_utterance_and_api_call_for_speaker(
                    "ASSISTANT", utterances, idx
                )
                if not self.opt["use_cumulative_api_calls"]:
                    r = tod.TodStructuredRound(
                        user_utt=user_utt,
                        api_call_machine=api_call,
                        api_resp_machine=api_resp,
                        sys_utt=sys_utt,
                    )
                else:
                    cum_api_call = self.process_call_for_cumlative_standalone_api(
                        api_call, cum_api_call
                    )
                    r = tod.TodStructuredRound(
                        user_utt=user_utt,
                        api_call_machine=cum_api_call if len(api_resp) > 0 else {},
                        api_resp_machine=api_resp if len(api_resp) > 0 else {},
                        sys_utt=sys_utt,
                    )

                rounds.append(r)
                if len(api_call) > 0:
                    goal_calls.append(api_call)

            episode = tod.TodStructuredEpisode(
                domain=tod.SerializationHelpers.inner_list_join(row["domain"]),
                api_schemas_machine=self._get_onto_list(ontologies, row["domain"]),
                goal_calls_machine=goal_calls,
                rounds=rounds,
                delex=self.opt.get("delex", False),
            )
            episodes.append(episode)
        return episodes

    def get_id_task_prefix(self):
        return "Taskmaster2"

    def _label_fold(self, chunks):
        return chunks.conversation_id.apply(self._h)

    def process_call_for_cumlative_standalone_api(self, new_call, cum_calls):
        if (
            len(new_call) > 0
            and len(cum_calls) > 0
            and new_call[tod.STANDARD_API_NAME_SLOT]
            != cum_calls[tod.STANDARD_API_NAME_SLOT]
        ):
            cum_calls = {}
        cum_calls.update(new_call)
        return cum_calls


class UserSimulatorTeacher(Taskmaster2Parser, tod_agents.TodUserSimulatorTeacher):
    pass


class SystemTeacher(Taskmaster2Parser, tod_agents.TodSystemTeacher):
    pass


class DefaultTeacher(SystemTeacher):
    pass
