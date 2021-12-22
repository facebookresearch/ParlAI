#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpers so we don't need to create agents all over.
"""

import parlai.core.tod.tod_agents as tod_agents
import parlai.core.tod.tod_core as tod_core

import os

API_DATABASE_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "standalone_api_file.pickle"
)


def episode_has_broken_api_turn(episode_idx, max_turns):
    """
    To generate predictably broken episodes (for calculating metrics)
    """
    return episode_idx % 2 == 1 and max_turns > 0


def turn_has_broken_api_call(round_idx, episode_idx):
    """
    To generate predictably broken turns (for calculating metrics)
    """
    return episode_idx % 2 == 1 and round_idx % 3 == 1


def make_api_call_machine(round_idx, episode_idx=0, use_broken_mock_api_calls=False):
    if round_idx == 0:
        return {}
    if use_broken_mock_api_calls:
        # Hack as a way to test metrics reporting in tod world script
        if turn_has_broken_api_call(round_idx, episode_idx):
            round_idx = -1 * round_idx
    return {tod_core.STANDARD_API_NAME_SLOT: f"name_{round_idx}", "in": round_idx}


def make_api_resp_machine(round_idx):
    if round_idx == 0:
        return {}
    return {"out": round_idx}


def make_api_schemas_machine(max_rounds):
    return [
        {
            tod_core.STANDARD_API_NAME_SLOT: f"name_{round_idx}",
            tod_core.STANDARD_REQUIRED_KEY: ["in"],
            tod_core.STANDARD_OPTIONAL_KEY: [],
        }
        for round_idx in range(1, max_rounds)
    ]


def make_goal_calls_machine(max_rounds):
    return [make_api_call_machine(x) for x in range(1, max_rounds)]


def get_rounds(episode_idx, max_rounds, use_broken_mock_api_calls=False):
    return [
        tod_core.TodStructuredRound(
            user_utt=f"user_utt_{episode_idx}_{round_idx}",
            api_call_machine=make_api_call_machine(
                round_idx, episode_idx, use_broken_mock_api_calls
            ),
            api_resp_machine=make_api_resp_machine(round_idx),
            sys_utt=f"sys_utt_{episode_idx}_{round_idx}",
        )
        for round_idx in range(max_rounds)
    ]


def get_round_utts(episode_idx, max_rounds, filter_utts=None):
    if max_rounds < 1:
        return []
    utts = [
        [
            f"{tod_core.STANDARD_USER_UTTERANCE}user_utt_{episode_idx}_0",
            tod_core.STANDARD_CALL,
            tod_core.STANDARD_RESP,
            f"{tod_core.STANDARD_SYSTEM_UTTERANCE}sys_utt_{episode_idx}_0",
        ]
    ]
    for i in range(1, max_rounds):
        utts.append(
            [
                f"{tod_core.STANDARD_USER_UTTERANCE}user_utt_{episode_idx}_{i}",
                f"{tod_core.STANDARD_CALL}api_name = name_{i} ; in = {i}",
                f"{tod_core.STANDARD_RESP}out = {i}",
                f"{tod_core.STANDARD_SYSTEM_UTTERANCE}sys_utt_{episode_idx}_{i}",
            ]
        )
    utts.append(
        [
            f"{tod_core.STANDARD_USER_UTTERANCE}{tod_core.STANDARD_DONE}",
            tod_core.STANDARD_CALL,
            tod_core.STANDARD_RESP,
            tod_core.STANDARD_SYSTEM_UTTERANCE,
        ]
    )
    if filter_utts is not None:
        utts = [
            [turn for i, turn in enumerate(round_data) if filter_utts[i]]
            for round_data in utts
        ]
    return utts


TEST_NUM_EPISODES_OPT_KEY = "test_num_episodes"
TEST_NUM_ROUNDS_OPT_KEY = "test_num_rounds"

# No api calls in this setup
EPISODE_SETUP__UTTERANCES_ONLY = {
    TEST_NUM_ROUNDS_OPT_KEY: 1,
    TEST_NUM_EPISODES_OPT_KEY: 1,
}

# Only one call, one goal, one api description in this setup
EPISODE_SETUP__SINGLE_API_CALL = {
    TEST_NUM_ROUNDS_OPT_KEY: 2,
    TEST_NUM_EPISODES_OPT_KEY: 1,
}
# Will start testing multiple api calls + schemas, multi-round logic
EPISODE_SETUP__MULTI_ROUND = {TEST_NUM_ROUNDS_OPT_KEY: 5, TEST_NUM_EPISODES_OPT_KEY: 1}

# Test that episode logic is correct
EPISODE_SETUP__MULTI_EPISODE = {
    TEST_NUM_ROUNDS_OPT_KEY: 5,
    TEST_NUM_EPISODES_OPT_KEY: 8,
}

# Test that episode + pesky-off-by-one batchinglogic is correct
EPISODE_SETUP__MULTI_EPISODE_BS = {
    TEST_NUM_ROUNDS_OPT_KEY: 5,
    TEST_NUM_EPISODES_OPT_KEY: 35,
}


class TestDataParser(tod_agents.TodStructuredDataParser):
    """
    Assume that when we init, we init w/ num of episodes + rounds as opts.
    """

    def __init__(self, opt, shared=None):
        opt["datafile"] = "DUMMY"
        self.fold = "DUMMY"
        # Following lines are only relevant in training the standalone api teacher
        if TEST_NUM_EPISODES_OPT_KEY not in opt:
            opt[TEST_NUM_EPISODES_OPT_KEY] = 35
        if TEST_NUM_ROUNDS_OPT_KEY not in opt:
            opt[TEST_NUM_ROUNDS_OPT_KEY] = 5
        super().__init__(opt, shared)

    def setup_episodes(self, _):
        result = []
        for ep_idx in range(0, self.opt[TEST_NUM_EPISODES_OPT_KEY]):
            result.append(
                tod_core.TodStructuredEpisode(
                    goal_calls_machine=[
                        make_api_call_machine(x)
                        for x in range(1, self.opt[TEST_NUM_ROUNDS_OPT_KEY])
                    ],
                    api_schemas_machine=make_api_schemas_machine(
                        self.opt[TEST_NUM_ROUNDS_OPT_KEY]
                    ),
                    rounds=get_rounds(
                        ep_idx,
                        self.opt[TEST_NUM_ROUNDS_OPT_KEY],
                        self.opt.get("use_broken_mock_api_calls", False),
                    ),
                )
            )
        return result

    def get_id_task_prefix(self):
        return "Test"


class SystemTeacher(TestDataParser, tod_agents.TodSystemTeacher):
    pass


class UserSimulatorTeacher(TestDataParser, tod_agents.TodUserSimulatorTeacher):
    pass


class StandaloneApiTeacher(TestDataParser, tod_agents.TodStandaloneApiTeacher):
    pass


class GoalAgent(TestDataParser, tod_agents.TodGoalAgent):
    pass


class ApiSchemaAgent(TestDataParser, tod_agents.TodApiSchemaAgent):
    pass


class SingleGoalAgent(TestDataParser, tod_agents.TodSingleGoalAgent):
    pass


class SingleApiSchemaAgent(TestDataParser, tod_agents.TodSingleApiSchemaAgent):
    pass


# Tested in tod world code
class UserUttAgent(TestDataParser, tod_agents.TodUserUttAgent):
    pass


# Tested in tod world code
class ApiCallAndSysUttAgent(TestDataParser, tod_agents.TodApiCallAndSysUttAgent):
    pass
