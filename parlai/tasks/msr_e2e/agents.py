#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
MSR-E2E implementation for ParlAI.

No official train/valid/test splits are available of public data, so we make our own.

We assume inform slots from the agent are API responses and request/inform slots from the user is an API call. It's not quite how things are supposed to work, but the `dialogue act` setup is not super well standardized within the dataset.
"""

from parlai.core.params import ParlaiParser
import copy
import os
import csv
from collections import Counter
from parlai.core.opt import Opt
import parlai.core.tod.tod_core as tod
from parlai.utils.misc import warn_once
from typing import Optional
from parlai.utils.data import DatatypeHelper
from parlai.utils.io import PathManager

import parlai.tasks.msr_e2e.build as build_
import parlai.core.tod.tod_agents as tod_agents


DOMAINS = ["movie", "restaurant", "taxi"]

# Just going to copy/paste these since it's faster than parsing 3 separate files
# They are in `system/src/deep_dialog/data_<domain>/<domain>_slots.txt` in the original data
SLOT_NAMES = {
    "movie": [
        "actor",
        "actress",
        "city",
        "closing",
        "critic_rating",
        "date",
        "schema",
        "distanceconstraints",
        "genre",
        "greeting",
        "implicit_value",
        "movie_series",
        "moviename",
        "mpaa_rating",
        "numberofpeople",
        "numberofkids",
        "taskcomplete",
        "other",
        "price",
        "seating",
        "starttime",
        "state",
        "theater",
        "theater_chain",
        "video_format",
        "zip",
        "result",
        "ticket",
        "mc_list",
    ],
    "restaurant": [
        "address",
        "atmosphere",
        "choice",
        "city",
        "closing",
        "cuisine",
        "date",
        "distanceconstraints",
        "dress_code",
        "food",
        "greeting",
        "implicit_value",
        "mealtype",
        "numberofpeople",
        "numberofkids",
        "occasion",
        "other",
        "personfullname",
        "phonenumber",
        "pricing",
        "rating",
        "restaurantname",
        "restauranttype",
        "seating",
        "starttime",
        "state",
        "zip",
        "result",
        "mc_list",
        "taskcomplete",
        "reservation",
    ],
    "taxi": [
        "car_type",
        "city",
        "closing",
        "date",
        "distanceconstraints",
        "dropoff_location",
        "greeting",
        "name",
        "numberofpeople",
        "other",
        "pickup_location",
        "dropoff_location_city",
        "pickup_location_city",
        "pickup_time",
        "state",
        "cost",
        "taxi_company",
        "mc_list",
        "taskcomplete",
        "taxi",
        "zip",
        "result",
        "mc_list",
    ],
}

SLOT_NAMES = {
    k: [{"api_name": k, "optArg": " | ".join(v)}] for k, v in SLOT_NAMES.items()
}


class MsrE2EParser(tod_agents.TodStructuredDataParser):
    """
    Abstract data loader.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.add_argument(
            "--msre2e-domains",
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
        self.dpath = os.path.join(opt["datapath"], "msr_e2e")
        if shared is None:
            warn_once("MsrE2E is a beta dataset, and format may significantly change.")
            build_.build(opt)
        super().__init__(opt, shared)

    def _load_data(self, fold, domains):
        chunks = []
        for section in domains:
            domain = []
            with PathManager.open(os.path.join(self.dpath, section + "_all.tsv")) as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader)
                lines = list(reader)
            episode = []
            prev_idx = 0
            for line in lines:
                data = {}
                data["id"] = line[0]
                data["speaker"] = line[3]
                data["text"] = line[4]
                data["dialogue_acts"] = line[5:]
                data["domain"] = section
                if prev_idx != data["id"]:
                    domain.append(episode)
                    episode = []
                    prev_idx = data["id"]
                episode.append(data)
            domain.append(episode)
            chunks.append(domain)
        # deterministic shuffle data for splits
        return DatatypeHelper.split_subset_data_by_fold(fold, chunks, 0.8, 0.1, 0.1)

    def _parse_dialogue_act(self, act, domain):
        act = (
            act.replace("inform(", "")
            .replace("request(", "")
            .replace("multiple_choice(", "")
        )
        act = act[:-1]

        args = act.split(";")
        result = {}
        for arg in args:
            params = arg.split("=")
            key = params[0]
            if (
                key == "other"
            ):  # This is all stuff the E2E model should be able to pick up on its own.
                continue
            if (
                len(params) == 1
            ):  # MSR_E2E has this as a "what explicit information do we want" slot, but it's not super consistent
                continue
            result[key] = "=".join(params[1:])
        if len(result) > 0:
            result[tod.STANDARD_API_NAME_SLOT] = domain
        return result

    def _get_utterance_and_api_call_for_speaker(self, speaker, utterances, idx):
        utts = []
        slots = {}
        while idx < len(utterances):
            here = utterances[idx]
            if here["speaker"] != speaker:
                break
            utts.append(here["text"])
            for act in utterances[idx]["dialogue_acts"]:
                if speaker == "agent" and not (
                    act.startswith("inform") or act.startswith("multiple_choice")
                ):
                    continue
                if speaker == "user" and not (
                    act.startswith("inform") or act.startswith("request")
                ):
                    continue
                slots.update(self._parse_dialogue_act(act, utterances[0]["domain"]))
            idx += 1
        return idx, "\n".join(utts), slots

    def setup_episodes(self, fold):
        """
        Parses into TodStructuredEpisode.
        """
        domains = self.opt.get("msre2e_domains", DOMAINS)
        chunks = self._load_data(fold, domains)
        domains_cnt = Counter()
        episodes = []
        for utterances in chunks:
            if len(utterances) < 1:
                continue
            domain = utterances[0]["domain"]
            domains_cnt[domain] += 1
            idx = 0
            rounds = []
            goal_calls = []
            if len(utterances) > 0 and utterances[0]["speaker"] == "agent":
                idx, sys_utt, api_resp = self._get_utterance_and_api_call_for_speaker(
                    "agent", utterances, idx
                )
                r = tod.TodStructuredRound(
                    user_utt=tod.CONST_SILENCE,
                    api_resp_machine=api_resp,
                    sys_utt=sys_utt,
                )
                rounds.append(r)

            cum_api_call = {}
            while idx < len(utterances):
                idx, user_utt, api_call = self._get_utterance_and_api_call_for_speaker(
                    "user", utterances, idx
                )
                idx, sys_utt, api_resp = self._get_utterance_and_api_call_for_speaker(
                    "agent", utterances, idx
                )
                if not self.opt["use_cumulative_api_calls"]:
                    r = tod.TodStructuredRound(
                        user_utt=user_utt,
                        api_call_machine=api_call,
                        api_resp_machine=api_resp,
                        sys_utt=sys_utt,
                    )
                else:
                    cum_api_call.update(api_call)
                    r = tod.TodStructuredRound(
                        user_utt=user_utt,
                        api_call_machine=copy.deepcopy(cum_api_call)
                        if len(api_resp) > 0
                        else {},
                        api_resp_machine=api_resp if len(api_resp) > 0 else {},
                        sys_utt=sys_utt,
                    )

                rounds.append(r)
                if len(api_call) > 0:
                    goal_calls.append(api_call)

            episode = tod.TodStructuredEpisode(
                domain=domain,
                api_schemas_machine=SLOT_NAMES[domain],
                goal_calls_machine=goal_calls,
                rounds=rounds,
                delex=self.opt.get("delex", False),
            )
            episodes.append(episode)
        return episodes

    def get_id_task_prefix(self):
        return "MsrE2E"

    def _label_fold(self, chunks):
        return chunks.conversation_id.apply(self._h)


class SystemTeacher(MsrE2EParser, tod_agents.TodSystemTeacher):
    pass


class UserSimulatorTeacher(MsrE2EParser, tod_agents.TodUserSimulatorTeacher):
    pass


class DefaultTeacher(SystemTeacher):
    pass
