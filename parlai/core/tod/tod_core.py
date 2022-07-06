#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Task Oriented Dialogue (TOD) enums and base classes.

This file defines standard tokens, classes for round and conversation structure, and a serialization class to aid in converting between these.

See `tod_agents.py` for usage of these classes to generate training data and `tod_world_script.py` for usage of these classes in simulated conversations.
"""
from enum import Enum
from typing import List, Dict
from dataclasses import dataclass, field
from collections.abc import Iterable
from parlai.utils.misc import warn_once

STANDARD_CALL = "APICALL: "
STANDARD_RESP = "APIRESP: "
STANDARD_SYSTEM_UTTERANCE = "SYSTEM: "
STANDARD_USER_UTTERANCE = "USER: "

STANDARD_GOAL = "GOAL: "
STANDARD_API_SCHEMAS = "APIS: "

STANDARD_API_NAME_SLOT = "api_name"
STANDARD_REQUIRED_KEY = "reqArg"
STANDARD_OPTIONAL_KEY = "optArg"
STANDARD_DONE = "[DONE]"

CONST_SILENCE = "__SILENCE__"


class TodAgentType(str, Enum):
    USER_UTT_AGENT = "user_utt_model"
    API_CALL_AGENT = "api_call_model"
    API_RESP_AGENT = "api_resp_model"
    SYSTEM_UTT_AGENT = "system_utt_model"
    API_SCHEMA_GROUNDING_AGENT = "api_schema_grounding_model"
    GOAL_GROUNDING_AGENT = "goal_grounding_model"


TOD_AGENT_TYPE_TO_PREFIX = {
    TodAgentType.USER_UTT_AGENT: STANDARD_USER_UTTERANCE,
    TodAgentType.API_CALL_AGENT: STANDARD_CALL,
    TodAgentType.API_RESP_AGENT: STANDARD_RESP,
    TodAgentType.SYSTEM_UTT_AGENT: STANDARD_SYSTEM_UTTERANCE,
    TodAgentType.API_SCHEMA_GROUNDING_AGENT: STANDARD_API_SCHEMAS,
    TodAgentType.GOAL_GROUNDING_AGENT: STANDARD_GOAL,
}


@dataclass
class TodStructuredRound:
    """
    Dataclass for rounds.

    After the first (grounding) turn, conversations in the TOD structure are rounds of
       1. User Utterance
       2. System API Call
       3. API Implementation API Response
       4. System Utterance

    This class hold that data.
    """

    # Variables set by those using this class
    user_utt: str = ""
    api_call_machine: Dict = field(
        default_factory=dict
    )  # Hashmap of slot keys and slot values. Note that STANDARD_API_NAME_SLOT (`api_name`) is expected to be one of the keys here when this is nonempty; simulation metrics wonky without
    api_resp_machine: Dict = field(default_factory=dict)
    sys_utt: str = ""
    extras: Dict = field(
        default_factory=dict
    )  # Grab bag for extra data. Not currently referenced in any TOD core code, but a convenient leaky abstraction for passing dataset-specific data between Parser classes and realized agents/teachers.

    # Variables derived by class
    api_call_utt: str = field(init=False)
    api_resp_utt: str = field(init=False)

    def __post_init__(self):
        self.api_call_utt = SerializationHelpers.api_dict_to_str(self.api_call_machine)
        self.api_resp_utt = SerializationHelpers.api_dict_to_str(self.api_resp_machine)
        if (
            len(self.api_call_machine) > 0
            and STANDARD_API_NAME_SLOT not in self.api_call_machine
        ):
            warn_once(
                f"{STANDARD_API_NAME_SLOT} missing when API Call present. This may cause issues for simulation metrics."
            )


@dataclass
class TodStructuredEpisode:
    """
    Dataclass for episode-level data.

    This holds the information for grounding turns (Goal calls, API Schemas), the rounds
    of User/System/API Implementation communications, as well as any extra metadata that
    is useful for the episode.
    """

    # Variables set by those using this class
    delex: bool = False  # Set to true and this class will handle delexicalizing call + response utterances based on API calls and responses exposed to this class.
    domain: str = ""  # self-explanatory
    api_schemas_machine: List[Dict[str, List]] = field(
        default_factory=list
    )  # Expected to be a List of Dicts with the API name, required arguments, and optional arguments (specified by consts at the top of this file) as keys
    goal_calls_machine: List[Dict[str, str]] = field(
        default_factory=list
    )  # Machine-formatted API calls
    rounds: List[TodStructuredRound] = field(default_factory=list)  # self explanatory
    extras: Dict = field(
        default_factory=dict
    )  # Grab bag for extra data. Not currently referenced in any TOD core code, but a convenient leaky abstraction for passing dataset-specific data between Parser classes and realized agents/teachers.

    # Variables derived by class
    api_schemas_utt: str = field(init=False)
    goal_calls_utt: str = field(init=False)

    def __post_init__(self):
        self.api_schemas_utt = SerializationHelpers.list_of_maps_to_str(
            self.api_schemas_machine
        )
        self.goal_calls_machine = [
            call for call in self.goal_calls_machine if len(call) > 0
        ]
        self.goal_calls_utt = SerializationHelpers.list_of_maps_to_str(
            self.goal_calls_machine
        )
        # Add a done turn at the end
        self.rounds.append(TodStructuredRound(user_utt=STANDARD_DONE))
        if self.delex:
            accum_slots = (
                {}
            )  # separate since some slot values change as we go. Use this for delex first
            cum_slots = self.get_all_slots()
            for r in self.rounds:
                accum_slots.update(r.api_call_machine)
                accum_slots.update(r.api_resp_machine)
                r.sys_utt = SerializationHelpers.delex(r.sys_utt, accum_slots)
                r.sys_utt = SerializationHelpers.delex(r.sys_utt, cum_slots)

    def get_all_slots(self):
        result = {}
        for r in self.rounds:
            result.update(r.api_call_machine)
            result.update(r.api_resp_machine)
        return result


class SerializationHelpers:
    @classmethod
    def delex(cls, text, slots):
        delex = text
        for slot, value in slots.items():
            if isinstance(value, str):
                delex = delex.replace(value, f"[{slot}]")
            else:
                for v in value:
                    delex = delex.replace(v, f"[{slot}]")
        return delex

    @classmethod
    def inner_list_join(cls, values):
        if isinstance(values, str):
            return values
        return ", ".join(sorted([str(v).strip() for v in values]))

    @classmethod
    def inner_list_split(cls, s):
        if len(s) < 1:
            return s
        if s[0] == "{":  # for case when we're json serializing a dict
            return s
        split = s.split(", ")
        if len(split) == 1:
            return split[0]
        return set(split)

    @classmethod
    def maybe_inner_list_join(cls, values):
        if type(values) is dict:
            return str(values)
        if (
            isinstance(values, str)
            or isinstance(values, int)
            or isinstance(values, float)
        ):
            return values
        elif isinstance(values, Iterable):
            return SerializationHelpers.inner_list_join(values)
        else:
            raise RuntimeError(
                f"invalid type of argument for maybe_inner_list_join: {values}; type {type(values)}"
            )

    @classmethod
    def api_dict_to_str(cls, apidict):
        """
        Used for API Calls and Responses -> Utterance.
        """

        return " ; ".join(
            f"{k} = {SerializationHelpers.maybe_inner_list_join(v)}"
            for k, v in sorted(apidict.items())
        )

    @classmethod
    def str_to_api_dict(cls, string):
        """
        Used for API Call and Response Utterances -> Dict.
        """
        slot_strs = string.split(" ; ")
        result = {}
        for slot_str in slot_strs:
            if " = " not in slot_str:
                continue
            name, value = slot_str.split(" = ", 1)
            name = name.strip()
            value = SerializationHelpers.inner_list_split(value.strip())
            result[name] = value
        return result

    @classmethod
    def outer_list_join(cls, s):
        return " | ".join(s)

    @classmethod
    def outer_list_split(cls, s):
        return s.split(" | ")

    @classmethod
    def str_to_list_of_maps(cls, s):
        return [
            SerializationHelpers.str_to_api_dict(x)
            for x in SerializationHelpers.outer_list_split(s)
        ]

    @classmethod
    def list_of_maps_to_str(cls, list_of_maps):
        return SerializationHelpers.outer_list_join(
            [SerializationHelpers.api_dict_to_str(m) for m in list_of_maps]
        )

    @classmethod
    def str_to_goals(cls, s):  # convenience
        return SerializationHelpers.str_to_list_of_maps(s)

    @classmethod
    def str_to_api_schemas(cls, s):  # convenience
        return SerializationHelpers.str_to_list_of_maps(s)
