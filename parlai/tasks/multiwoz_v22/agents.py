#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
implementation for ParlAI.
"""

from parlai.core.params import ParlaiParser
import copy
import os
import pandas as pd
from parlai.core.opt import Opt
import parlai.core.tod.tod_core as tod
import json
from typing import Optional
from parlai.utils.data import DatatypeHelper
from parlai.utils.io import PathManager

import parlai.tasks.multiwoz_v22.build as build_
import parlai.core.tod.tod_agents as tod_agents


DOMAINS = [
    "attraction",
    "bus",
    "hospital",
    "hotel",
    "police",
    "restaurant",
    "taxi",
    "train",
]

WELL_FORMATTED_DOMAINS = ["attraction", "bus", "hotel", "restaurant", "train", "taxi"]


class MultiwozV22Parser(tod_agents.TodStructuredDataParser):
    """
    Abstract data loader for Multiwoz V2.2 into TOD structured data format.

    Multiwoz 2.2 has 'find' and 'book' as the only intents.

    For API calls, we look for turns that are not 'NONE' `active_intents` in the USER's turn state. We then filter these for whether or not the SYSTSEM has actually made an api call by looking in the dialogue act of the SYSTEM turn.
    * For 'find' intents, we make an API call if it does an "Inform" or gives a "NoOffer". We look in the corresponding `.db` file to return the relevant information.
    * For 'book' intents, we make an API call if the SYSTEM's dialogue act includes  booking and then offer the slots/values of that key as the API response.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        group = parser.add_argument_group("Multiwoz args")
        group.add_argument(
            "--well-formatted-domains-only",
            type=bool,
            default=True,
            help="Some of the domains in Multiwoz are not super well formatted. Use only the well formatted ones.",
        )
        group.add_argument(
            "--dialogue-id",
            type=str,
            default="",
            help="If non-empty, filters for a particular dialogue id",
        )
        return super().add_cmdline_args(parser, partial_opt)

    def __init__(self, opt: Opt, shared=None):
        self.fold = DatatypeHelper.fold(opt["datatype"])
        opt["datafile"] = self.fold
        self.dpath = os.path.join(opt["datapath"], "multiwoz_v22")
        build_.build(opt)
        self.last_call = {}
        super().__init__(opt, shared)

    def load_schemas(self):
        with PathManager.open(os.path.join(self.dpath, "schema.json")) as f:
            raw = json.load(f)
        result = {}
        for service in raw:
            domain = service["service_name"]
            prefix_end_idx = len(domain) + 1
            all_slots = set([x["name"][prefix_end_idx:] for x in service["slots"]])

            for intent in service["intents"]:
                call_name = intent["name"]
                result[call_name] = {tod.STANDARD_API_NAME_SLOT: call_name}
                req_slots = set([x[prefix_end_idx:] for x in intent["required_slots"]])
                if len(req_slots) > 0:
                    result[call_name][tod.STANDARD_REQUIRED_KEY] = list(req_slots)

                # Not fully trusting the original schema data...
                optional_slots = set(
                    [x[prefix_end_idx:] for x in intent["optional_slots"].keys()]
                )
                optional_slots = optional_slots | all_slots
                optional_slots = optional_slots - req_slots
                if len(optional_slots) > 0:
                    result[call_name][tod.STANDARD_OPTIONAL_KEY] = list(optional_slots)

            if domain == "police":  # Multiwoz 2.2 only lists "police"
                result["find_police"] = {
                    tod.STANDARD_OPTIONAL_KEY: list(all_slots),
                    tod.STANDARD_API_NAME_SLOT: "find_police",
                }
            if (
                domain == "taxi"
            ):  # Multiwoz 2.2 has "book taxi" in the schema but it's "find taxi" in the data...
                result["find_taxi"] = copy.deepcopy(result["book_taxi"])
                result["find_taxi"][tod.STANDARD_API_NAME_SLOT] = "find_taxi"
        return result

    def load_dbs(self):
        dbs = {}
        for key in DOMAINS:
            if (
                key == "hospital"
            ):  # has funky extra format, so we're gonna deal with it manually.
                with PathManager.open(
                    os.path.join(self.dpath, "db", key + "_db.json")
                ) as f:
                    file_lines = f.readlines()
                hospital_address_lines = file_lines[1:4]
                partial = [
                    x.replace("#", "").strip().lower().split(":")
                    for x in hospital_address_lines
                ]
                self.hospital_address = {x[0]: x[1] for x in partial}
                self.hospital_department_details = json.loads("".join(file_lines[6:]))
                continue
            if (
                key == "taxi"
            ):  # Taxi domain is funky and the db for it is just response slot options.
                continue
            with PathManager.open(
                os.path.join(self.dpath, "db", key + "_db.json")
            ) as f:
                blob = json.load(f)
                for i, entry in enumerate(blob):
                    cased = {}
                    for slot_name in entry:
                        cased[slot_name.lower().replace(" ", "")] = entry[slot_name]
                    blob[i] = cased
                dbs[key] = pd.DataFrame.from_dict(blob)

        return dbs

    def load_chunks(self, fold):
        if fold == "valid":
            fold = "dev"  # change name to match file structure
        for path in PathManager.ls(os.path.join(self.dpath, fold)):
            with PathManager.open(os.path.join(self.dpath, fold, path)) as f:
                blob = json.load(f)
                for convo in blob:
                    yield convo

    def _get_find_api_response(self, intent, raw_slots, sys_dialog_act):
        """
        Get an API response out of the lookup databases.
        """
        domain = ""
        for cand in DOMAINS:
            if cand in intent:
                domain = cand
        if domain == "taxi":  # handle separately cause funky
            for action in sys_dialog_act:
                if action == "Taxi-Inform":
                    return {x[0]: x[1] for x in sys_dialog_act[action]}
            return {domain: domain}  # too much work to do this right...
        if domain == "hospital":  # handle separately cause funky
            res = self.hospital_address
            if "hospital-department" in raw_slots:
                for blob in self.hospital_department_details:
                    if blob["department"] in raw_slots["hospital-department"]:
                        res[blob["department"]] = blob
            return res
        slots = {}
        for raw_key in raw_slots:
            key = raw_key[len(domain + "-") :]
            slots[key] = raw_slots[raw_key]
        for action in sys_dialog_act:
            if "Recommend" in action:
                add_slots = {}
                for x in sys_dialog_act[action]:
                    name = x[0]
                    val = x[1]
                    if self._slot_in_schema(name, intent):
                        if name not in add_slots:
                            add_slots[name] = []
                        add_slots[name].append(val)
                for key in add_slots:
                    slots[key] = add_slots[key]

        find = self.dbs[domain]
        for slot, values in slots.items():
            if slot == "arriveby":
                condition = find[slot] < values[0]
            elif slot == "leaveat":
                condition = find[slot] > values[0]
            else:
                condition = find[slot].isin(values)

            find = find[condition]

        filtered = self.dbs[domain].iloc[find.index]
        count = len(filtered.index)
        if count == 0:
            return {}
        blob = filtered.head(1).to_dict('records')

        results = blob[0]
        results["COUNT"] = count
        return results

    def _slot_in_schema(self, slot, intent):
        return slot in self.schemas[intent].get(
            tod.STANDARD_OPTIONAL_KEY, []
        ) or slot in self.schemas[intent].get(tod.STANDARD_REQUIRED_KEY, [])

    def _get_round(self, dialogue_id, raw_episode, turn_id):
        """
        Parse to TodStructuredRound.

        Assume User turn first.
        """
        user_turn = raw_episode[turn_id]
        if user_turn["speaker"] != "USER":
            raise RuntimeError(
                f"Got non-user turn when it should have been in {dialogue_id}; turn id {turn_id}"
            )
        sys_turn = raw_episode[turn_id + 1]
        sys_dialog_act = self.dialog_acts[dialogue_id][str(turn_id + 1)]["dialog_act"]
        if sys_turn["speaker"] != "SYSTEM":
            raise RuntimeError(
                f"Got non-system turn when it should have been in {dialogue_id}; turn id {turn_id}"
            )
        frames = user_turn.get("frames", [])
        call = {}
        resp = {}
        for frame in frames:
            if frame.get("state", {}).get("active_intent", "NONE") != "NONE":
                intent = frame["state"]["active_intent"]
                domain = frame["service"]
                maybe_call_raw = copy.deepcopy(frame["state"]["slot_values"])
                maybe_call = {}
                truncate_length = len(domain) + 1
                for key in maybe_call_raw:
                    maybe_call[key[truncate_length:]] = maybe_call_raw[key][0]
                maybe_call[tod.STANDARD_API_NAME_SLOT] = intent
                if "find" in intent:
                    for key in sys_dialog_act:
                        if "Inform" in key or "NoOffer" in key:
                            # Gotta check to make sure if it's inform, that it's about the right topic
                            if "Inform" in key:
                                valid = True
                                slots = [x[0] for x in sys_dialog_act[key]]
                                for slot in slots:
                                    valid &= self._slot_in_schema(slot, intent) | (
                                        slot == "choice"
                                    )
                                if not valid:
                                    continue

                            call = maybe_call
                            call_key = str(call)
                            if call_key not in self.call_response_cache:
                                resp = self._get_find_api_response(
                                    intent,
                                    frame["state"]["slot_values"],
                                    sys_dialog_act,
                                )
                                self.call_response_cache[call_key] = resp
                            else:
                                resp = self.call_response_cache[call_key]

                elif "book" in intent:
                    for key in sys_dialog_act:
                        if "Book" in key:  # and "Inform" not in key:
                            resp = {x[0]: x[1] for x in sys_dialog_act[key]}
                            call = maybe_call
        if call == self.last_call:
            call = {}
            resp = {}
        if len(call) > 0:
            self.last_call = call
        return (
            call,
            tod.TodStructuredRound(
                user_utt=user_turn["utterance"],
                api_call_machine=call,
                api_resp_machine=resp,
                sys_utt=sys_turn["utterance"],
            ),
        )

    def _get_schemas_for_goal_calls(self, goals):
        result = []
        seen = set()
        for goal in goals:
            call_name = goal[tod.STANDARD_API_NAME_SLOT]
            if call_name not in seen:
                result.append(self.schemas[call_name])
                seen.add(call_name)
        return result

    def setup_episodes(self, fold):
        """
        Parses into TodStructuredEpisode.
        """
        self.schemas = self.load_schemas()
        cache_path = os.path.join(self.dpath, f"{fold}_call_response_cache.json")

        if PathManager.exists(cache_path):
            with PathManager.open(cache_path, 'r') as f:
                self.call_response_cache = json.load(f)
            self.dbs = None
        else:
            self.call_response_cache = {}
            self.dbs = self.load_dbs()

        with PathManager.open(os.path.join(self.dpath, "dialog_acts.json")) as f:
            self.dialog_acts = json.load(f)

        chunks = self.load_chunks(fold)

        episodes = []
        for raw_episode in chunks:
            domains = raw_episode["services"]

            if self.opt.get("dialogue_id", "") != "":
                if raw_episode["dialogue_id"] != self.opt["dialogue_id"]:
                    continue

            skip = False  # need to skip outer for loop while in `for domains` inner for loop
            if self.opt.get("well_formatted_domains_only", True):
                if len(domains) == 0:
                    skip = True
                for domain in domains:
                    if domain not in WELL_FORMATTED_DOMAINS:
                        skip = True
            if skip:
                continue

            turn_id = 0  # matching naming in the `dialogues` files.
            turns = raw_episode["turns"]
            rounds = []
            goal_calls = []

            while turn_id < len(turns):
                goal, r = self._get_round(raw_episode['dialogue_id'], turns, turn_id)
                turn_id += 2
                rounds.append(r)

                if len(goal) > 0:
                    goal_calls.append(goal)

            episode = tod.TodStructuredEpisode(
                domain=tod.SerializationHelpers.inner_list_join(domains),
                api_schemas_machine=self._get_schemas_for_goal_calls(goal_calls),
                goal_calls_machine=goal_calls,
                rounds=rounds,
            )
            episodes.append(episode)

        with PathManager.open(cache_path, 'w') as f:
            json.dump(self.call_response_cache, f)

        return episodes

    def get_id_task_prefix(self):
        return "MultiwozV22"


class UserSimulatorTeacher(MultiwozV22Parser, tod_agents.TodUserSimulatorTeacher):
    pass


class SystemTeacher(MultiwozV22Parser, tod_agents.TodSystemTeacher):
    pass


class StandaloneApiTeacher(MultiwozV22Parser, tod_agents.TodStandaloneApiTeacher):
    pass


class SingleGoalAgent(MultiwozV22Parser, tod_agents.TodSingleGoalAgent):
    pass


class SingleApiSchemaAgent(MultiwozV22Parser, tod_agents.TodSingleApiSchemaAgent):
    pass


class DefaultTeacher(SystemTeacher):
    pass
