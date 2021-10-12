#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Task Oriented Dialogue (TOD) enums and base classes.
"""
from enum import Enum
from typing import List, Dict
from dataclasses import dataclass, field
from collections.abc import Iterable

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

API_CALL_CALL_DOES_NOT_EXIST = "ERR_CALL_DNE"


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
    """

    user_utt: str = ""
    api_call_utt: str = field(init=False)
    api_resp_utt: str = field(init=False)
    sys_utt: str = ""
    api_call_machine: Dict = field(default_factory=dict)
    api_resp_machine: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.api_call_utt = SerializationHelpers.api_dict_to_str(self.api_call_machine)
        self.api_resp_utt = SerializationHelpers.api_dict_to_str(self.api_resp_machine)


@dataclass
class TodStructuredEpisode:
    """
    Dataclass for episode-level data.
    """

    delex: bool = False  # Set to true for delexicalized call + response utterances.
    domain: str = ""
    all_domains: [str] = field(default_factory=list)  # list of strings
    api_schemas_machine: List[Dict[str, List]] = field(default_factory=list)
    goal_calls_machine: List[Dict[str, str]] = field(default_factory=list)
    rounds: List[TodStructuredRound] = field(default_factory=list)

    api_schemas_utt: str = field(init=False)
    goal_calls_utt: str = field(init=False)
    extras: Dict = field(default_factory=dict)

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
        return ", ".join(sorted([v.strip() for v in values]))

    @classmethod
    def inner_list_split(cls, s):
        return s.split(", ")

    @classmethod
    def maybe_inner_list_join(cls, values):
        if isinstance(values, str) or isinstance(values, int):
            return values
        elif isinstance(values, Iterable):
            return SerializationHelpers.inner_list_join(values)
        else:
            raise RuntimeError("invalid type of argument for maybe_inner_list_join")

    @classmethod
    def api_dict_to_str(cls, apidict):
        return " ; ".join(
            f"{k} = {SerializationHelpers.maybe_inner_list_join(v)}"
            for k, v in sorted(apidict.items())
        )

    @classmethod
    def str_to_api_dict(cls, string):
        slot_strs = string.split(" ; ")
        result = {}
        for slot_str in slot_strs:
            if " = " not in slot_str:
                continue
            name, value = slot_str.split(" = ", 1)
            name = name.strip()
            value = value.strip()
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
    def str_to_goals(cls, s):
        return SerializationHelpers.str_to_list_of_maps(s)

    @classmethod
    def str_to_api_schemas(cls, s):
        api_call_lists = SerializationHelpers.str_to_list_of_maps(s)

        def further_split(api_call):
            res = {
                k: SerializationHelpers.inner_list_split(v) for k, v in api_call.items()
            }
            if STANDARD_API_NAME_SLOT in res:
                res[STANDARD_API_NAME_SLOT] = res[STANDARD_API_NAME_SLOT][0]
            return res

        return [further_split(x) for x in api_call_lists]
