#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Multiwoz 2.2 Dataset implementation for ParlAI.
"""

import copy
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

import parlai.core.tod.tod_agents as tod_agents
import parlai.core.tod.tod_core as tod
import parlai.tasks.multiwoz_v22.build as build_
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.data import DatatypeHelper
from parlai.utils.io import PathManager

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

DATA_LEN = {"train": 17, "dev": 2, "test": 2}

SEED = 42


def fold_size(fold):
    return DATA_LEN[fold]


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


class MultiWOZv22DSTTeacher(MultiwozV22Parser, tod_agents.TodUserSimulatorTeacher):
    """
    This Teacher is responsible for performing the task of Dialogue State Tracking.

    It can be used to evaluate LM on JGA (Joint Goal Accuracy) metric (as shown in
    [SimpleTOD](https://arxiv.org/abs/2005.00796) and
    [Soloist](https://arxiv.org/abs/2005.05298)).
    """

    BELIEF_STATE_DELIM = " ; "

    domains = [
        "attraction",
        "hotel",
        "hospital",
        "restaurant",
        "police",
        "taxi",
        "train",
    ]

    named_entity_slots = {
        "attraction--name",
        "restaurant--name",
        "hotel--name",
        "bus--departure",
        "bus--destination",
        "taxi--departure",
        "taxi--destination",
        "train--departure",
    }

    rng = np.random.RandomState(SEED)

    def __init__(self, opt: Opt, shared=None, *args, **kwargs):
        self.opt = opt
        self.fold = opt["datatype"].split(":")[0]
        opt["datafile"] = self.fold
        self.dpath = os.path.join(opt["datapath"], "multiwoz_v22")
        self.id = "multiwoz_v22"

        if shared is None:
            build_.build(opt)
        super().__init__(opt, shared)

    def _load_data(self, fold):
        dataset_fold = "dev" if fold == "valid" else fold
        fold_path = os.path.join(self.dpath, dataset_fold)
        dialogs = []
        for file_id in range(1, fold_size(dataset_fold) + 1):
            filename = os.path.join(fold_path, f"dialogues_{file_id:03d}.json")
            with PathManager.open(filename, "r") as f:
                dialogs += json.load(f)
        return dialogs

    def _get_curr_belief_states(self, turn):
        belief_states = []
        for frame in turn["frames"]:
            if "state" in frame:
                if "slot_values" in frame["state"]:
                    for domain_slot_type in frame["state"]["slot_values"]:
                        for slot_value in frame["state"]["slot_values"][
                            domain_slot_type
                        ]:
                            domain, slot_type = domain_slot_type.split("-")
                            belief_state = f"{domain} {slot_type} {slot_value.lower()}"
                            belief_states.append(belief_state)
        return list(set(belief_states))

    def _extract_slot_from_string(self, slots_string):
        """
        Either ground truth or generated result should be in the format: "dom slot_type
        slot_val, dom slot_type slot_val, ..., dom slot_type slot_val," and this
        function would reformat the string into list:

        ["dom--slot_type--slot_val", ... ]
        """

        slots_list = []
        per_domain_slot_lists = {}
        named_entity_slot_lists = []

        # split according to ";"
        str_split = slots_string.split(self.BELIEF_STATE_DELIM)

        if str_split[-1] == "":
            str_split = str_split[:-1]

        str_split = [slot.strip() for slot in str_split]

        for slot_ in str_split:
            slot = slot_.split()
            if len(slot) > 2 and slot[0] in self.domains:
                domain = slot[0]
                slot_type = slot[1]
                slot_val = " ".join(slot[2:])
                if not slot_val == "dontcare":
                    slots_list.append(domain + "--" + slot_type + "--" + slot_val)
                if domain in per_domain_slot_lists:
                    per_domain_slot_lists[domain].add(slot_type + "--" + slot_val)
                else:
                    per_domain_slot_lists[domain] = {slot_type + "--" + slot_val}
                if domain + "--" + slot_type in self.named_entity_slots:
                    named_entity_slot_lists.append(
                        domain + "--" + slot_type + "--" + slot_val
                    )
        return slots_list, per_domain_slot_lists, named_entity_slot_lists

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        """
        for dialog state tracking, we compute the joint goal accuracy, which is the
        percentage of the turns where the model correctly and precisely predicts all
        slots(domain, slot_type, slot_value).
        """
        resp = model_response.get("text")
        if not resp:
            return

        # extract ground truth from labels
        (
            slots_truth,
            slots_truth_per_domain,
            slots_truth_named_entity,
        ) = self._extract_slot_from_string(labels[0])

        # extract generated slots from model_response
        (
            slots_pred,
            slots_pred_per_domain,
            slots_pred_named_entity,
        ) = self._extract_slot_from_string(resp)

        for gt_slot in slots_truth:
            self.metrics.add("all/slot_r", AverageMetric(gt_slot in slots_pred))
            curr_domain = gt_slot.split("--")[0]
            self.metrics.add(
                f"{curr_domain}/slot_r", AverageMetric(gt_slot in slots_pred)
            )

        for gt_slot in slots_pred_named_entity:
            self.metrics.add(
                "hallucination", AverageMetric(gt_slot not in slots_truth_named_entity)
            )

        for predicted_slot in slots_pred:
            self.metrics.add("all/slot_p", AverageMetric(predicted_slot in slots_truth))
            curr_domain = predicted_slot.split("--")[0]
            self.metrics.add(
                f"{curr_domain}/slot_p", AverageMetric(predicted_slot in slots_truth)
            )

        self.metrics.add("jga", AverageMetric(set(slots_truth) == set(slots_pred)))
        self.metrics.add(
            "named_entities/jga",
            AverageMetric(
                set(slots_truth_named_entity) == set(slots_pred_named_entity)
            ),
        )
        for gt_slot in slots_truth_named_entity:
            self.metrics.add("all_ne/slot_r", AverageMetric(gt_slot in slots_pred))
            curr_domain = gt_slot.split("--")[0]
            self.metrics.add(
                f"{curr_domain}_ne/slot_r", AverageMetric(gt_slot in slots_pred)
            )
        for predicted_slot in slots_pred_named_entity:
            self.metrics.add(
                "all_ne/slot_p", AverageMetric(predicted_slot in slots_truth)
            )
            curr_domain = predicted_slot.split("--")[0]
            self.metrics.add(
                f"{curr_domain}_ne/slot_p", AverageMetric(predicted_slot in slots_truth)
            )

        for domain in slots_truth_per_domain:
            if domain in slots_pred_per_domain:
                self.metrics.add(
                    f"{domain}/jga",
                    AverageMetric(
                        slots_truth_per_domain[domain] == slots_pred_per_domain[domain]
                    ),
                )

    def setup_data(self, fold):
        dialogs = self._load_data(fold)
        examples = []
        for dialog in dialogs:
            context = []
            for turn in dialog["turns"]:
                curr_turn = turn["utterance"].lower()
                curr_speaker = (
                    "<user>" if turn["speaker"].lower() == "user" else "<system>"
                )
                curr_context = f"{curr_speaker} {curr_turn}"
                context.append(curr_context)
                cum_belief_states = self._get_curr_belief_states(turn)
                if curr_speaker == "<user>":
                    examples.append(
                        {
                            "dialogue_id": dialog["dialogue_id"],
                            "turn_num": turn["turn_id"],
                            "text": " ".join(context),
                            "labels": self.BELIEF_STATE_DELIM.join(
                                set(cum_belief_states)
                            ),
                        }
                    )

        self.rng.shuffle(examples)
        for example in examples:
            yield example, True


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
