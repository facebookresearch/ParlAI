#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests different (more complicated) slot metrics.
"""

import unittest

import copy
import parlai.core.tod.tod_core as tod_core
import parlai.core.tod.tod_test_utils.agents_and_teachers as aat


class TestTodAgentsAndTeachersBase(unittest.TestCase):
    def setup_agent_or_teacher(self, class_type, round_opt, opt):
        full_opts = {**round_opt, **opt}
        full_opts["datatype"] = "DUMMY"
        full_opts["datafile"] = "DUMMY"
        full_opts["episodes_randomization_seed"] = 0
        return class_type(full_opts)

    def dump_single_utt_per_episode_agent_text(self, class_type, round_opt, opt):
        agent = self.setup_agent_or_teacher(class_type, round_opt, opt)
        result = []
        while not agent.epoch_done():
            result.append(agent.act()["text"])
            agent.reset()
        return result

    def dump_teacher_text(self, class_type, round_opt, opt):
        """
        Array where [episode_idx][turn_idx][text=0,label=1]
        """
        teacher = self.setup_agent_or_teacher(class_type, round_opt, opt)
        data = []
        here = []
        for x, new in teacher.setup_data("dummy"):
            if new and len(here) > 0:
                data.append(copy.deepcopy(here))
                here = []
            here.append([x["text"], x["label"]])
        if len(here) > 0:
            data.append(here)
        return data

    def _test_roundDataCorrect(self):
        self._test_roundDataCorrect_helper(aat.EPISODE_SETUP__UTTERANCES_ONLY)
        self._test_roundDataCorrect_helper(aat.EPISODE_SETUP__SINGLE_API_CALL)
        self._test_roundDataCorrect_helper(aat.EPISODE_SETUP__MULTI_ROUND)
        self._test_roundDataCorrect_helper(aat.EPISODE_SETUP__MULTI_EPISODE)


class TestSystemTeacher(TestTodAgentsAndTeachersBase):
    def test_apiDescriptions_with_yesApiDescriptions(self):
        values = self.dump_teacher_text(
            aat.SystemTeacher,
            aat.EPISODE_SETUP__SINGLE_API_CALL,
            {"api_descriptions": True},
        )
        self.assertEqual(
            values[0][0][0],
            "APIS: "
            + tod_core.SerializationHelpers.list_of_maps_to_str(
                aat.make_api_descriptions_machine(2)
            ),
        )

    def test_apiDescriptions_with_noApiDescriptions(self):
        values = self.dump_teacher_text(
            aat.SystemTeacher,
            aat.EPISODE_SETUP__SINGLE_API_CALL,
            {"api_descriptions": False},
        )
        self.assertEqual(values[0][0][0], "APIS: ")

    def _test_roundDataCorrect_helper(self, config):
        max_rounds = config[aat.TEST_NUM_ROUNDS_OPT_KEY]
        values = self.dump_teacher_text(aat.SystemTeacher, config, {})
        for episode_idx, episode in enumerate(values):
            utts = aat.get_round_utts(episode_idx, max_rounds)
            comp = []
            for utt in utts:
                comp.append([utt[0], utt[1]])
                comp.append([utt[2], utt[3]])
            # Skip context turn cause we check it above
            self.assertEqual(episode[1:], comp)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestUserTeacher(TestTodAgentsAndTeachersBase):
    def _test_roundDataCorrect_helper(self, config):
        max_rounds = config[aat.TEST_NUM_ROUNDS_OPT_KEY]
        values = self.dump_teacher_text(aat.UserSimulatorTeacher, config, {})
        for episode_idx, episode in enumerate(values):
            utts = aat.get_round_utts(episode_idx, max_rounds)
            comp = []
            comp.append(
                [
                    "GOAL: "
                    + tod_core.SerializationHelpers.list_of_maps_to_str(
                        aat.make_goal_calls_machine(max_rounds)
                    ),
                    utts[0][0],
                ]
            )
            last_sys = utts[0][3]
            for i in range(1, len(utts)):
                comp.append([last_sys, utts[i][0]])
                last_sys = utts[i][3]
            self.assertEqual(episode, comp)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestGoalAgent(TestTodAgentsAndTeachersBase):
    def _test_roundDataCorrect_helper(self, config):
        max_rounds = config[aat.TEST_NUM_ROUNDS_OPT_KEY]
        max_episodes = config[aat.TEST_NUM_EPISODES_OPT_KEY]
        values = self.dump_single_utt_per_episode_agent_text(aat.GoalAgent, config, {})

        goal_text = [
            "GOAL: "
            + tod_core.SerializationHelpers.list_of_maps_to_str(
                aat.make_goal_calls_machine(max_rounds)
            )
            for _ in range(max_episodes)
        ]

        self.assertEqual(values, goal_text)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestApiDescriptionAgent(TestTodAgentsAndTeachersBase):
    def _test_roundDataCorrect_helper(self, config):
        max_rounds = config[aat.TEST_NUM_ROUNDS_OPT_KEY]
        max_episodes = config[aat.TEST_NUM_EPISODES_OPT_KEY]
        values = self.dump_single_utt_per_episode_agent_text(
            aat.ApiDescriptionAgent, config, {}
        )

        apis_texts = [
            "APIS: "
            + tod_core.SerializationHelpers.list_of_maps_to_str(
                aat.make_api_descriptions_machine(max_rounds)
            )
            for _ in range(max_episodes)
        ]

        self.assertEqual(values, apis_texts)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestSingleGoalAgent(TestTodAgentsAndTeachersBase):
    def _test_roundDataCorrect_helper(self, config):
        max_rounds = config[aat.TEST_NUM_ROUNDS_OPT_KEY]
        max_episodes = config[aat.TEST_NUM_EPISODES_OPT_KEY]
        values = self.dump_single_utt_per_episode_agent_text(
            aat.SingleGoalAgent, config, {}
        )

        goal_text = []
        for _ in range(max_episodes):
            goals = aat.make_goal_calls_machine(max_rounds)
            for x in goals:
                goal_text.append(
                    "GOAL: " + tod_core.SerializationHelpers.list_of_maps_to_str([x])
                )

        self.assertEqual(values, goal_text)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestSingleApiDescriptionAgent(TestTodAgentsAndTeachersBase):
    def _test_roundDataCorrect_helper(self, config):
        max_rounds = config[aat.TEST_NUM_ROUNDS_OPT_KEY]
        max_episodes = config[aat.TEST_NUM_EPISODES_OPT_KEY]
        values = self.dump_single_utt_per_episode_agent_text(
            aat.SingleApiDescriptionAgent, config, {}
        )

        apis_text = []
        for _ in range(max_episodes):
            apis = aat.make_api_descriptions_machine(max_rounds)
            for x in apis:
                apis_text.append(
                    "APIS: " + tod_core.SerializationHelpers.list_of_maps_to_str([x])
                )
        self.assertEqual(values, apis_text)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestSingleGoalWithSingleApiDescriptionAgent(TestTodAgentsAndTeachersBase):
    """
    Make sure the SingleGoal + SingleApiDescription agents correspond.
    """

    def _test_roundDataCorrect_helper(self, config):
        goals = self.dump_single_utt_per_episode_agent_text(
            aat.SingleGoalAgent, config, {}
        )
        apis = self.dump_single_utt_per_episode_agent_text(
            aat.SingleApiDescriptionAgent, config, {}
        )

        for i in range(len(goals)):
            goal = tod_core.SerializationHelpers.str_to_goals(goals[i][len("GOALS:") :])
            api = tod_core.SerializationHelpers.str_to_api_descriptions(
                apis[i][len("APIS:") :]
            )
            self.assertEqual(
                goal[0].get("api_name", None), api[0].get("api_name", None)
            )

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


if __name__ == "__main__":
    unittest.main()
