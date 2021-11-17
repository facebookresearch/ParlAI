#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Taskmaster-1 implementation for ParlAI.

Note that we have conversations structured both in the "TOD" format as well as those
from prior.
"""

from parlai.core.params import ParlaiParser
import os
import pandas as pd
from parlai.core.opt import Opt
import parlai.core.tod.tod_core as tod
from typing import Optional
from parlai.utils.data import DatatypeHelper
from parlai.utils.io import PathManager

import parlai.tasks.taskmaster.build as build_
import parlai.core.tod.tod_agents as tod_agents

# Following is for legacy format
from parlai.core.teachers import FixedDialogTeacher
from . import tm_utils
import json


################### TOD Conversation format

SILENCE_TOKEN = "__SILENCE__"

# Faster to copy/paste this than parse a json file
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
                (idx, sys_utt, _) = self._get_utterance_and_slots_for_speaker(
                    "ASSISTANT", utterances, idx
                )

                turns = self._get_turns_from_parsed(SILENCE_TOKEN, {}, {}, sys_utt)
                for t in turns:
                    rounds.append(t)

            while idx < len(utterances):
                (idx, user_utt, user_slots) = self._get_utterance_and_slots_for_speaker(
                    "USER", utterances, idx
                )
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


class SystemTeacher(Taskmaster1Parser, tod_agents.TodSystemTeacher):
    pass


class UserSimulatorTeacher(Taskmaster1Parser, tod_agents.TodUserSimulatorTeacher):
    pass


############ Legacy defined teachers


class SelfDialogueTeacher(FixedDialogTeacher):
    """
    Teacher for written two-person dialogues with labels being responses for the
    previous statement.

    The data is traversed twice (doubled), once for modelling USER replies and once for
    modelling ASSISTANT replies.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        opt['fn'] = "self-dialogs.json"

        if shared and 'convos' in shared:
            # another instance was set up already, just reference its data
            self.convos = shared['convos']
            self.ep_cheat_sheet = shared['ep_cheat_sheet']
            self.num_ex = shared['num_ex']
        else:
            # need to set up data from scratch
            self.ep_cheat_sheet = {}  # Stores imp. info. about each episode
            data_path = tm_utils._path(opt)
            self.num_ex = 0
            self._setup_data(data_path, opt)

        self.reset()

    def _setup_data(self, data_path, opt):
        print('loading: ' + data_path)
        with PathManager.open(data_path) as data_file:
            self.convos = json.load(data_file)
        # Pre-processing
        convos_update = []
        for convo in self.convos:
            conversation = convo['utterances']
            # Filter out single greet messages
            if len(conversation) > 1:
                self.ep_cheat_sheet[
                    len(self.ep_cheat_sheet)
                ] = tm_utils.gen_ep_cheatsheet(conversation)
                curr_cheatsheet = self.ep_cheat_sheet[len(self.ep_cheat_sheet) - 1]
                self.num_ex += (
                    curr_cheatsheet[tm_utils.USER_NUM_EX]
                    + curr_cheatsheet[tm_utils.ASSIS_NUM_EX]
                )
                convos_update += [conversation]
        self.convos = convos_update

    def num_examples(self):
        return self.num_ex

    def num_episodes(self):
        # For two passes over the data: Once to teach USER and once to teach ASSISTANT
        return len(self.convos) * 2

    def get(self, episode_idx, entry_idx):
        conversation = self.convos[episode_idx % len(self.convos)]
        if episode_idx < len(self.convos):
            # USER then ASSISTANT mode [First pass]
            ep_done = (
                entry_idx * 2
                == self.ep_cheat_sheet[episode_idx][tm_utils.LAST_USER_IDX]
            )
            predecessor = conversation[entry_idx * 2]['text']
            successor = conversation[entry_idx * 2 + 1]['text']
        else:
            # ASSISTANT then USER mode [Second pass]
            ep_done = (
                entry_idx * 2 + 1
                == self.ep_cheat_sheet[episode_idx % len(self.convos)][
                    tm_utils.LAST_ASSISTANT_IDX
                ]
            )
            predecessor = conversation[entry_idx * 2 + 1]['text']
            successor = conversation[entry_idx * 2 + 2]['text']

        action = {
            'id': self.id,
            'text': predecessor,
            'episode_done': ep_done,
            'labels': [successor],
        }

        return action


class WozDialogueTeacher(FixedDialogTeacher):
    """
    Teacher for spoken two-person dialogues with labels being responses for the previous
    statement.

    The data is traversed twice (doubled), once for modelling USER replies and once for
    modelling ASSISTANT replies.
    """

    def __init__(self, opt, shared=None):
        opt['fn'] = "woz-dialogs.json"
        super().__init__(opt)

        if shared and 'convos' in shared:
            # another instance was set up already, just reference its data
            self.convos = shared['convos']
            self.episode_map = shared['episode_map']
            self.ep_cheat_sheet = shared['ep_cheat_sheet']
            self.num_ex = shared['num_ex']
        else:
            # need to set up data from scratch
            self.ep_cheat_sheet = {}  # Stores imp. info. about each episode

            # Not all episodes have relevant examples for both USER and ASSISTANT
            # episode_map keeps track of which episode index is useful for which speaker
            # Need to do this otherwise might end up with a situation where we cannot
            # return anything in action
            self.episode_map = {}
            self.episode_map["U"] = {}
            self.episode_map["A"] = {}
            self.num_ex = 0
            data_path = tm_utils._path(opt)
            self._setup_data(data_path, opt)

        self.reset()

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Corrupt-Example-Arguments')
        agent.add_argument(
            '--exclude-invalid-data',
            type='bool',
            default=True,
            help='Whether to include corrupt examples in the data',
        )
        return parser

    def _setup_data(self, data_path, opt):
        print('loading: ' + data_path)
        with PathManager.open(data_path) as data_file:
            self.convos = json.load(data_file)
        # Pre-processing
        convos_update = []
        for convo in self.convos:
            conversation, corrupted = tm_utils.smoothen_convo(convo, opt)
            # Filter out single greet messages and corrupted examples
            if len(conversation) > 1 and not corrupted:
                actual_ep_idx = len(self.ep_cheat_sheet)
                self.ep_cheat_sheet[actual_ep_idx] = tm_utils.gen_ep_cheatsheet(
                    conversation
                )
                curr_cheatsheet = self.ep_cheat_sheet[len(self.ep_cheat_sheet) - 1]
                # calc number of examples (done here to prevent double counting if done later)
                self.num_ex += (
                    curr_cheatsheet[tm_utils.USER_NUM_EX]
                    + curr_cheatsheet[tm_utils.ASSIS_NUM_EX]
                )
                # User example exists
                if curr_cheatsheet[tm_utils.USER_NUM_EX] != 0:
                    u_idx = len(self.episode_map["U"])
                    self.episode_map["U"][u_idx] = actual_ep_idx
                # Assistant example exists
                if curr_cheatsheet[tm_utils.ASSIS_NUM_EX] != 0:
                    a_idx = len(self.episode_map["A"])
                    self.episode_map["A"][a_idx] = actual_ep_idx
                convos_update += [conversation]
        self.convos = convos_update

    def num_examples(self):
        return self.num_ex

    def num_episodes(self):
        # For two passes over the data: Once to teach USER and once to teach ASSISTANT
        return len(self.episode_map["U"]) + len(self.episode_map["A"])

    def get(self, episode_idx, entry_idx):
        if episode_idx < len(self.episode_map["U"]):
            # USER then ASSISTANT mode [First pass]
            true_idx = self.episode_map["U"][episode_idx]
            conversation = self.convos[true_idx]
            convo_cheat_sheet = self.ep_cheat_sheet[true_idx]
            first_entry_idx, last_entry_idx = (
                convo_cheat_sheet[tm_utils.FIRST_USER_IDX],
                convo_cheat_sheet[tm_utils.LAST_USER_IDX],
            )
        else:
            # ASSISTANT then USER mode [Second pass]
            episode_idx -= len(
                self.episode_map["U"]
            )  # Didn't use '%' because the two maybe unequal in length
            true_idx = self.episode_map["A"][episode_idx]
            conversation = self.convos[true_idx]
            convo_cheat_sheet = self.ep_cheat_sheet[true_idx]
            first_entry_idx, last_entry_idx = (
                convo_cheat_sheet[tm_utils.FIRST_ASSISTANT_IDX],
                convo_cheat_sheet[tm_utils.LAST_ASSISTANT_IDX],
            )

        starts_at_odd = first_entry_idx % 2 != 0
        if starts_at_odd:
            predecessor = conversation[entry_idx * 2 + 1]['text']
            successor = conversation[entry_idx * 2 + 2]['text']
            ep_done = entry_idx * 2 + 1 == last_entry_idx
        else:
            predecessor = conversation[entry_idx * 2]['text']
            successor = conversation[entry_idx * 2 + 1]['text']
            ep_done = entry_idx * 2 == last_entry_idx

        action = {
            'id': self.id,
            'text': predecessor,
            'episode_done': ep_done,
            'labels': [successor],
        }

        return action


class SelfDialogueSegmentTeacher(FixedDialogTeacher):
    """
    Teacher for written two-person dialogues with labels being relevant/useful parts in
    the input sentence.

    The different datatypes of the labels within the data have also been encoded as
    `label_types`
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        opt['fn'] = "self-dialogs.json"

        if shared and 'convos' in shared:
            # another instance was set up already, just reference its data
            self.convos = shared['convos']
            self.num_ex = shared['num_ex']
        else:
            # need to set up data from scratch
            data_path = tm_utils._path(opt)
            self.num_ex = 0
            self._setup_data(data_path, opt)
        self.reset()

    def get(self, episode_idx, entry_idx):
        conversation = self.convos[episode_idx]
        conv_len = len(conversation['utterances'])
        utterance = conversation['utterances'][entry_idx]['text']

        # Check if episode is complete
        ep_done = entry_idx == conv_len - 1
        action = {'id': self.id, 'text': utterance, 'episode_done': ep_done}

        # Setup Labels as "text" from segments
        action['labels'] = []
        action['label_types'] = []
        segments = conversation['utterances'][entry_idx]["segments"]
        for segment in segments:
            action['labels'] += [segment["text"]]
            tmp = []
            for annot in segment["annotations"]:
                tmp += [annot["name"]]
            action['label_types'] += [tmp]

        return action

    def num_examples(self):
        return self.num_ex

    def num_episodes(self):
        return len(self.convos)

    def _setup_data(self, data_path, opt):
        print('loading: ' + data_path)
        with PathManager.open(data_path) as data_file:
            self.convos = json.load(data_file)

        # Filter out instances which do not have "segment" in them
        convos_updated = []
        for convo in self.convos:
            updated_dialog = []
            for i in range(len(convo['utterances'])):
                if "segments" in convo['utterances'][i]:
                    updated_dialog += [convo['utterances'][i]]
            convo['utterances'] = updated_dialog
            if convo['utterances']:
                convos_updated += [convo]
                self.num_ex += len(convo['utterances'])
        self.convos = convos_updated


class DefaultTeacher(SelfDialogueTeacher):
    pass
