#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Taskmaster-1 implementation for ParlAI.

There are official splits for Taskmaster-1, but since we've already got the TM2/TM3 code
written and it's faster to copy/paste, we're just not going to worry about using their
official splits for now...
"""

from parlai.core.params import ParlaiParser
import os
import pandas as pd
from parlai.core.opt import Opt
import parlai.core.tod.tod_core as tod
from parlai.utils.misc import warn_once
from typing import Optional
from parlai.utils.data import DatatypeHelper
from parlai.utils.io import PathManager

import parlai.tasks.taskmaster.build as build_
import parlai.core.tod.tod_agents_and_teachers as tod_agents
import parlai.core.tod.tod_agents_and_teachers as tod_teachers

SILENCE_TOKEN = "__SILENCE__"

ONTOLOGY = {
    "uber": {
        "id": "uber_lyft",
        "vertical": "ride_booking",
        "required": ["location.from", "location.to", "type.ride", "num.people"],
        "optional": [
            "price.estimate",
            "duration.estimate",
            "time.pickup",
            "time.dropoff",
        ],
    },
    "movie": {
        "id": "movie_ticket",
        "vertical": "ticket_booking",
        "required": [
            "name.movie",
            "name.theater",
            "num.tickets",
            "time.start",
            "location.theater",
            "price.ticket",
        ],
        "optional": ["type.screening", "time.end", "time.duration"],
    },
    "restaurant": {
        "id": "restaurant_reservation",
        "vertical": "reservation",
        "required": [
            "name.restaurant",
            "name.reservation",
            "num.guests",
            "time.reservation",
        ],
        "optional": ["type.seating", "location.restaurant"],
    },
    "coffee": {
        "id": "coffee_ordering",
        "vertical": "coffee_order",
        "required": ["location.store", "name.drink", "size.drink"],
        "optional": ["num.drink", "type.milk", "preference"],
    },
    "pizza": {
        "id": "pizza_ordering",
        "vertical": "pizza_order",
        "required": ["name.store", "name.pizza", "size.pizza"],
        "optional": ["type.topping", "type.crust", "preference", "location.store"],
    },
    "auto": {
        "id": "auto_repair",
        "vertical": "appointment",
        "required": ["name.store", "name.customer", "date.appt", "time.appt"],
        "optional": ["reason.appt", "name.vehicle", "year.vehicle", "location.store"],
    },
}


class Taskmaster1Parser(tod_agents.TodStructuredDataParser):
    """
    Abstract data loader.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        return parser

    def __init__(self, opt: Opt, shared=None):
        self.fold = DatatypeHelper.fold(opt["datatype"])
        opt["datafile"] = self.fold
        self.dpath = os.path.join(opt["datapath"], "taskmaster-1")
        if shared is None:
            warn_once(
                "Taskmaster1 is a beta dataset, and format may significantly change."
            )
            build_.build(opt)
        super().__init__(opt, shared)

    def _load_data(self, fold):
        chunks = []
        with PathManager.open(os.path.join(self.dpath, f"self-dialogs.json")) as f:
            subset = pd.read_json(f)
            chunks.append(subset)
        with PathManager.open(os.path.join(self.dpath, f"woz-dialogs.json")) as f:
            subset = pd.read_json(f)
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
        return chunks, ONTOLOGY

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

    def _get_utterance_and_slots_for_speaker(self, speaker, utterances, idx):
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

    def _parse_to_api_schema(self, raw):
        """
        NOTE: Format of ontology in this is different from TM2 + TM3. Need to figure out which is relevant for the domain.
        """
        result = {}
        for key, val in raw.items():
            here = {}
            here[tod.STANDARD_API_NAME_SLOT] = val["id"]
            here[tod.STANDARD_REQUIRED_KEY] = val.get("required", [])
            here[tod.STANDARD_OPTIONAL_KEY] = val.get("optional", [])
            result[key] = here
        return result

    def _get_turns_from_parsed(self, user_utt, api_calls, api_resps, sys_utt):
        result = [
            tod.TodStructuredRound(
                user_utt=user_utt,
                api_call_machine=api_calls,
                api_resp_machine=api_resps,
                sys_utt=sys_utt,
            )
        ]
        return result

    def setup_episodes(self, fold):
        """
        Parses into TodStructuredEpisode.
        """
        chunks, api_schema_raw = self._load_data(fold)
        api_schemas_machine = self._parse_to_api_schema(api_schema_raw)
        episodes = []
        for _, row in chunks.iterrows():
            utterances = row["utterances"][:]
            if not all(
                [
                    x.get("speaker") == "ASSISTANT" or x.get("speaker") == "USER"
                    for x in utterances
                ]
            ):
                # there's an example or two that causes things to infinite loop. >.>
                continue
            idx = 0
            rounds = []
            goal_calls = []
            if len(utterances) > 0 and utterances[0]["speaker"] == "ASSISTANT":
                (idx, sys_utt, _,) = self._get_utterance_and_slots_for_speaker(
                    "ASSISTANT", utterances, idx
                )

                turns = self._get_turns_from_parsed(SILENCE_TOKEN, {}, {}, sys_utt)
                for t in turns:
                    rounds.append(t)

            while idx < len(utterances):
                (
                    idx,
                    user_utt,
                    user_slots,
                ) = self._get_utterance_and_slots_for_speaker("USER", utterances, idx)
                (
                    idx,
                    sys_utt,
                    system_slots,
                ) = self._get_utterance_and_slots_for_speaker(
                    "ASSISTANT", utterances, idx
                )
                # The annotations in this dataset don't make sense as api responses but... we'll just roll.
                turns = self._get_turns_from_parsed(
                    user_utt, user_slots, system_slots, sys_utt
                )
                for t in turns:
                    rounds.append(t)
            apis = []
            for candidate_api in api_schemas_machine:
                if candidate_api in row["instruction_id"]:
                    apis.append(api_schemas_machine[candidate_api])
            episode = tod.TodStructuredEpisode(
                api_schemas_machine=apis,
                goal_calls_machine=goal_calls,
                rounds=rounds,
                delex=self.opt.get("delex", False),
            )
            episodes.append(episode)
        return episodes

    def get_id_task_prefix(self):
        return "Taskmaster1"

    def _label_fold(self, chunks):
        return chunks.conversation_id.apply(self._h)


class SystemTeacher(Taskmaster1Parser, tod_teachers.SystemTeacher):
    pass


class UserSimulatorTeacher(Taskmaster1Parser, tod_teachers.UserSimulatorTeacher):
    pass


class StandaloneApiTeacher(Taskmaster1Parser, tod_teachers.TodStandaloneApiTeacher):
    pass


class GoalAgent(Taskmaster1Parser, tod_agents.TodGoalAgent):
    pass


class ApiSchemaAgent(Taskmaster1Parser, tod_agents.TodApiSchemaAgent):
    pass


class UserUttAgent(Taskmaster1Parser, tod_agents.TodUserUttAgent):
    pass


class ApiCallAndSysUttAgent(Taskmaster1Parser, tod_agents.TodApiCallAndSysUttAgent):
    pass


class DefaultTeacher(SystemTeacher):
    pass
