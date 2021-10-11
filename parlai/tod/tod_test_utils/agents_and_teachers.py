#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpers so we don't need to create agents all over.
"""

import parlai.tod.tod_agents as tod_agents
import parlai.tod.tod_teachers as tod_teachers
import parlai.tod.tod_core as tod_core


def make_api_call_machine(round_idx):
    if round_idx == 0:
        return {}
    return {tod_core.STANDARD_API_NAME_SLOT: f"name_{round_idx}", "in": round_idx}


def make_api_resp_machine(round_idx):
    if round_idx == 0:
        return {}
    return {"out": round_idx}


def make_api_descriptions_machine(max_rounds):
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


def get_rounds(episode_idx, max_rounds):
    return [
        tod_core.TodStructuredRound(
            user_utt=f"user_utt_{episode_idx}_{round_idx}",
            api_call_machine=make_api_call_machine(round_idx),
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
            f"USER: user_utt_{episode_idx}_0",
            "APICALL: ",
            "APIRESP: ",
            f"SYSTEM: sys_utt_{episode_idx}_0",
        ]
    ]
    for i in range(1, max_rounds):
        utts.append(
            [
                f"USER: user_utt_{episode_idx}_{i}",
                f"APICALL: api_name = name_{i} ; in = {i}",
                f"APIRESP: out = {i}",
                f"SYSTEM: sys_utt_{episode_idx}_{i}",
            ]
        )
    utts.append(
        [
            "USER: [DONE]",
            "APICALL: ",
            "APIRESP: ",
            "SYSTEM: ",
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

# No one call, one goal, one api desscription in this setup
EPISODE_SETUP__SINGLE_API_CALL = {
    TEST_NUM_ROUNDS_OPT_KEY: 2,
    TEST_NUM_EPISODES_OPT_KEY: 1,
}
# Will start testing multiple api calls + descriptions, multi-round logic
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


class TestDataParser(tod_agents.TodStructuredDataAgent):
    """
    Assume that when we init, we init w/ num of episodes + rounds as opts.
    """

    def __init__(self, opt, shared=None):
        opt["datafile"] = "DUMMY"
        opt["episodes_randomization_seed"] = -1  # no random.
        self.fold = "DUMMY"
        # Following lines are only reelvant in training the standalone api teacher
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
                    api_descriptions_machine=make_api_descriptions_machine(
                        self.opt[TEST_NUM_ROUNDS_OPT_KEY]
                    ),
                    rounds=get_rounds(ep_idx, self.opt[TEST_NUM_ROUNDS_OPT_KEY]),
                )
            )
        return result

    def get_id_task_prefix(self):
        return "Test"


class SystemTeacher(TestDataParser, tod_teachers.SystemTeacher):
    pass


class UserSimulatorTeacher(TestDataParser, tod_teachers.UserSimulatorTeacher):
    pass


class StandaloneApiTeacher(TestDataParser, tod_teachers.TodStandaloneApiTeacher):
    pass


class GoalAgent(TestDataParser, tod_agents.TodGoalAgent):
    pass


class ApiDescriptionAgent(TestDataParser, tod_agents.TodApiDescriptionAgent):
    pass


class SingleGoalAgent(TestDataParser, tod_agents.TodSingleGoalAgent):
    pass


class SingleApiDescriptionAgent(
    TestDataParser, tod_agents.TodSingleApiDescriptionAgent
):
    pass


# Tested in tod world code
class UserUttAgent(TestDataParser, tod_agents.TodUserUttAgent):
    pass


# Tested in tod world code
class ApiCallAndSysUttAgent(TestDataParser, tod_agents.TodApiCallAndSysUttAgent):
    pass
