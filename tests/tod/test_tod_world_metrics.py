#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test world metrics + world metrics handlers against dummy conversations.
"""

import unittest

from parlai.core.tod.tod_core import (
    STANDARD_API_NAME_SLOT,
    STANDARD_REQUIRED_KEY,
    STANDARD_OPTIONAL_KEY,
    TodStructuredRound,
    TodStructuredEpisode,
    TodAgentType,
    TOD_AGENT_TYPE_TO_PREFIX,
)
from parlai.core.tod.world_metrics import TodMetrics
from parlai.core.tod.world_metrics_handlers import METRICS_HANDLER_CLASSES_TEST_REGISTRY

# Ignore lint on following line; want to have registered classes show up for tests
import projects.tod_simulator.world_metrics.extended_world_metrics  # noqa: F401

GOAL__SINGLE_ONE_KEY = [{STANDARD_API_NAME_SLOT: "name", "a": "1"}]
GOAL__SINGLE_THREE_KEYS = [
    {STANDARD_API_NAME_SLOT: "name", "a": "1", "b": "2", "c": "3"}
]
GOAL__HARD = [
    {
        STANDARD_API_NAME_SLOT: "otherName",
        "w": "1",
        "x": "2",
        "y": "3",
        "z": "will_be_missing",
        "diff": "right",
    }
]

API_CALL__NO_API_NAME_SLOT = {"random": "blah"}
API_CALL__API_NAME_DNE = {STANDARD_API_NAME_SLOT: "not_an_api_name"}
API_CALL__VALID_NAME_BUT_EMPTY = {STANDARD_API_NAME_SLOT: "name"}
API_CALL__SINGLE_ONE_KEY = GOAL__SINGLE_ONE_KEY[0]
API_CALL__SINGLE_ONE_KEY_WITH_OPT = {**GOAL__SINGLE_ONE_KEY[0], **{"c": "3"}}
API_CALL__SINGLE_ONE_KEY_WITH_OPT_AND_NONVALID = {
    **GOAL__SINGLE_ONE_KEY[0],
    **{"c": "3", "nonExistent": "blah"},
}
API_CALL__FUNKY_AGAINST_HARD = {
    STANDARD_API_NAME_SLOT: "otherName",
    "w": "1",
    "x": "2",
    "y": "3",
    "diff": "wrong",
}

API_SCHEMA__ONE_CALL_ONE_REQ_MATCH_ONE_KEY = [
    {
        STANDARD_API_NAME_SLOT: "name",
        STANDARD_REQUIRED_KEY: ["a"],
        STANDARD_OPTIONAL_KEY: [],
    }
]

API_SCHEMA__ONE_CALL_MATCH_THREE_KEYS = [
    {
        STANDARD_API_NAME_SLOT: "name",
        STANDARD_REQUIRED_KEY: ["a"],
        STANDARD_OPTIONAL_KEY: ["b", "c", "d"],
    }
]

API_SCHEMA__ONE_CALL_HARD = [
    {
        STANDARD_API_NAME_SLOT: "otherName",
        STANDARD_REQUIRED_KEY: ["w", "x"],
        STANDARD_OPTIONAL_KEY: ["y", "z", "diff"],
    }
]


class TodMetricsTestHelper:
    """
    Given a synthetic intermediate converesation, calculates the metrics for said
    conversation.
    """

    def __init__(self, e: TodStructuredEpisode):
        self.m = TodMetrics()
        self.m.handlers = [
            x() for x in METRICS_HANDLER_CLASSES_TEST_REGISTRY
        ]  # run on ALL
        self.e = e

    def _process(self, t: TodAgentType, text: str):
        self.m.handle_message({"text": f"{TOD_AGENT_TYPE_TO_PREFIX[t]}{text}"}, t)

    def run(self):
        self._process(TodAgentType.API_SCHEMA_GROUNDING_AGENT, self.e.api_schemas_utt)
        self._process(TodAgentType.GOAL_GROUNDING_AGENT, self.e.goal_calls_utt)

        for r in self.e.rounds:
            self._process(TodAgentType.USER_UTT_AGENT, r.user_utt)
            self._process(TodAgentType.API_CALL_AGENT, r.api_call_utt)
            self._process(TodAgentType.API_RESP_AGENT, r.api_resp_utt)
            self._process(TodAgentType.SYSTEM_UTT_AGENT, r.sys_utt)

        self.m.episode_reset()

    def report(self):
        return self.m.report()


class TestApiGoalHitMetricsHandler(unittest.TestCase):
    def __helper(self, api_schemas_machine, goal_calls_machine, single_turn_api_call):
        e = TodStructuredEpisode(
            api_schemas_machine=api_schemas_machine,
            goal_calls_machine=goal_calls_machine,
            rounds=[TodStructuredRound(api_call_machine=single_turn_api_call)],
        )
        helper = TodMetricsTestHelper(e)
        helper.run()
        result = helper.report()
        return result

    def test_one_goal_only_req(self):
        result = self.__helper(
            api_schemas_machine=API_SCHEMA__ONE_CALL_ONE_REQ_MATCH_ONE_KEY,
            goal_calls_machine=GOAL__SINGLE_ONE_KEY,
            single_turn_api_call=API_CALL__SINGLE_ONE_KEY,
        )
        self.assertAlmostEqual(result["all_goals_hit"], 1)
        self.assertAlmostEqual(result["all_goals_hit_turn_count"], 1)
        self.assertAlmostEqual(result["all_goals_fractional_hit"], 1)
        self.assertAlmostEqual(result["all_goals_slot_precision"], 1)
        self.assertAlmostEqual(result["all_goals_slot_recall"], 1)

        self.assertAlmostEqual(result["req_goals_hit"], 1)
        self.assertAlmostEqual(result["req_goals_hit_turn_count"], 1)
        self.assertAlmostEqual(result["req_goals_fractional_hit"], 1)
        self.assertAlmostEqual(result["req_goals_slot_precision"], 1)
        self.assertAlmostEqual(result["req_goals_slot_recall"], 1)

    def test_one_goal_api_name_missing_slots(self):
        result = self.__helper(
            api_schemas_machine=API_SCHEMA__ONE_CALL_ONE_REQ_MATCH_ONE_KEY,
            goal_calls_machine=GOAL__SINGLE_ONE_KEY,
            single_turn_api_call=API_CALL__VALID_NAME_BUT_EMPTY,
        )
        self.assertAlmostEqual(result["all_goals_hit"], 0)
        self.assertAlmostEqual(result["all_goals_hit_turn_count"], 0)
        self.assertAlmostEqual(result["all_goals_fractional_hit"], 0)
        self.assertAlmostEqual(result["all_goals_slot_precision"], 1)  # api_name
        self.assertAlmostEqual(result["all_goals_slot_recall"], 0.5)

        self.assertAlmostEqual(result["req_goals_hit"], 0)
        self.assertAlmostEqual(result["req_goals_hit_turn_count"], 0)
        self.assertAlmostEqual(result["req_goals_fractional_hit"], 0)
        self.assertAlmostEqual(result["req_goals_slot_precision"], 1)
        self.assertAlmostEqual(result["req_goals_slot_recall"], 0.5)

    def test_one_goal_with_opts(self):
        result = self.__helper(
            api_schemas_machine=API_SCHEMA__ONE_CALL_MATCH_THREE_KEYS,
            goal_calls_machine=GOAL__SINGLE_THREE_KEYS,
            single_turn_api_call=API_CALL__SINGLE_ONE_KEY,
        )
        self.assertAlmostEqual(result["all_goals_hit"], 0)
        self.assertAlmostEqual(result["all_goals_hit_turn_count"], 0)
        self.assertAlmostEqual(result["all_goals_fractional_hit"], 0)
        self.assertAlmostEqual(result["all_goals_slot_precision"], 1)
        self.assertAlmostEqual(result["all_goals_slot_recall"], 0.5)

        self.assertAlmostEqual(result["req_goals_hit"], 1)
        self.assertAlmostEqual(result["req_goals_hit_turn_count"], 1)
        self.assertAlmostEqual(result["req_goals_fractional_hit"], 1)
        self.assertAlmostEqual(result["req_goals_slot_precision"], 1)
        self.assertAlmostEqual(result["req_goals_slot_recall"], 1)

    def test_hard_case(self):
        result = self.__helper(
            api_schemas_machine=API_SCHEMA__ONE_CALL_HARD,
            goal_calls_machine=GOAL__HARD,
            single_turn_api_call=API_CALL__FUNKY_AGAINST_HARD,
        )
        self.assertAlmostEqual(result["all_goals_hit"], 0)
        self.assertAlmostEqual(result["all_goals_hit_turn_count"], 0)
        self.assertAlmostEqual(result["all_goals_fractional_hit"], 0)
        self.assertAlmostEqual(result["all_goals_slot_precision"], 0.8)
        self.assertAlmostEqual(result["all_goals_slot_recall"], 2.0 / 3.0)

        self.assertAlmostEqual(result["req_goals_hit"], 1)
        self.assertAlmostEqual(result["req_goals_hit_turn_count"], 1)
        self.assertAlmostEqual(result["req_goals_fractional_hit"], 1)
        self.assertAlmostEqual(result["req_goals_slot_precision"], 0.6)
        self.assertAlmostEqual(result["req_goals_slot_recall"], 1)


class TestApiCallMalformedMetricsHandler(unittest.TestCase):
    def __helper(self, single_turn_api_call):
        e = TodStructuredEpisode(
            api_schemas_machine=API_SCHEMA__ONE_CALL_MATCH_THREE_KEYS,
            rounds=[TodStructuredRound(api_call_machine=single_turn_api_call)],
        )
        helper = TodMetricsTestHelper(e)
        helper.run()
        return helper.report()

    def test_no_api_name_slot(self):
        result = self.__helper(API_CALL__NO_API_NAME_SLOT)
        self.assertEqual(result["apiCall_wellFormed"], 0)
        self.assertEqual(result["apiCall_hasSlotsButNoApiNameSlot_count"], 1)

    def test_api_name_DNE(self):
        result = self.__helper(API_CALL__API_NAME_DNE)
        self.assertEqual(result["apiCall_wellFormed"], 0)
        self.assertEqual(result["apiCall_methodDNE_count"], 1)

    def test_missing_required_slot(self):
        result = self.__helper(API_CALL__VALID_NAME_BUT_EMPTY)
        self.assertEqual(result["apiCall_wellFormed"], 0)
        self.assertEqual(result["apiCall_missingRequiredSlot_count"], 1)

    def test_has_single_required_slot(self):
        result = self.__helper(API_CALL__SINGLE_ONE_KEY)
        self.assertEqual(result["apiCall_wellFormed"], 1)
        self.assertEqual(result["apiCall_wellFormed_count"], 1)

    def test_has_valid_optional_slot(self):
        result = self.__helper(API_CALL__SINGLE_ONE_KEY_WITH_OPT)
        self.assertEqual(result["apiCall_wellFormed"], 1)
        self.assertEqual(result["apiCall_wellFormed_count"], 1)

    def test_has_invalid_extra_slots(self):
        result = self.__helper(API_CALL__SINGLE_ONE_KEY_WITH_OPT_AND_NONVALID)
        self.assertEqual(result["apiCall_wellFormed"], 0)
        self.assertEqual(result["apiCall_hasExtraParams_count"], 1)


if __name__ == "__main__":
    unittest.main()
