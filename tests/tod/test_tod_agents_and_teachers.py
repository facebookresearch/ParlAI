#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests teachers + agent implementations, assuming parser to conversations format has
already been done and teachers/agents already created.

`test_agents.py` includes functions for generating the raw data used in this file as
well as the data parser.
"""

import unittest

import copy
import parlai.core.tod.tod_core as tod_core
import parlai.core.tod.tod_test_utils.test_agents as test_agents


class TestTodAgentsAndTeachersBase(unittest.TestCase):
    """
    Base class with convenience functions for setting up agents, dumping text, etc.
    """

    def setup_agent_or_teacher(self, class_type, round_opt, opt):
        full_opts = {**round_opt, **opt}
        full_opts["datatype"] = "DUMMY"
        full_opts["datafile"] = "DUMMY"
        full_opts["episodes_randomization_seed"] = -1  # no random here
        return class_type(full_opts)

    def dump_single_utt_per_episode_agent_text(self, class_type, round_opt, opt):
        """
        Continuously dumps data from an agent until it's done.
        """
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
        """
        Convenience function that runs on different episode setups.

        Prefix with `_` since not all tests necessarily need this
        """
        self._test_roundDataCorrect_helper(test_agents.EPISODE_SETUP__UTTERANCES_ONLY)
        self._test_roundDataCorrect_helper(test_agents.EPISODE_SETUP__SINGLE_API_CALL)
        self._test_roundDataCorrect_helper(test_agents.EPISODE_SETUP__MULTI_ROUND)
        self._test_roundDataCorrect_helper(test_agents.EPISODE_SETUP__MULTI_EPISODE)

    def _test_roundDataCorrect_helper(self, config):
        """
        Implement this in downstream classes to define what is "correct" for a round (Ie
        checking serialization data for a given class vs only checking utterances)
        """
        raise RuntimeError("Not implemented")


class TestSystemTeacher(TestTodAgentsAndTeachersBase):
    def test_apiSchemas_with_yesApiSchemas(self):
        """
        Tests to make sure that data from first turn is correct when we include API
        Schemas.
        """
        values = self.dump_teacher_text(
            test_agents.SystemTeacher,
            test_agents.EPISODE_SETUP__SINGLE_API_CALL,
            {"api_schemas": True},
        )
        self.assertEqual(
            values[0][0][0],
            "APIS: "
            + tod_core.SerializationHelpers.list_of_maps_to_str(
                test_agents.make_api_schemas_machine(2)
            ),
        )

    def test_apiSchemas_with_noApiSchemas(self):
        """
        Tests to make sure that data from first turn is correct when we do not include
        API Schemas.
        """
        values = self.dump_teacher_text(
            test_agents.SystemTeacher,
            test_agents.EPISODE_SETUP__SINGLE_API_CALL,
            {"api_schemas": False},
        )
        self.assertEqual(values[0][0][0], "APIS: ")

    def _test_roundDataCorrect_helper(self, config):
        max_rounds = config[test_agents.TEST_NUM_ROUNDS_OPT_KEY]
        values = self.dump_teacher_text(test_agents.SystemTeacher, config, {})
        for episode_idx, episode in enumerate(values):
            utts = test_agents.get_round_utts(episode_idx, max_rounds)
            comp = []
            for utt in utts:
                comp.append([utt[0], utt[1]])
                comp.append([utt[2], utt[3]])
            # Skip grounding turn cause we check it in the other teachers
            self.assertEqual(episode[1:], comp)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestUserTeacher(TestTodAgentsAndTeachersBase):
    def _test_roundDataCorrect_helper(self, config):
        """
        Make sure that all of the User teacher data is correct relative to ground truth,
        including grounding turn.
        """
        max_rounds = config[test_agents.TEST_NUM_ROUNDS_OPT_KEY]
        values = self.dump_teacher_text(test_agents.UserSimulatorTeacher, config, {})
        for episode_idx, episode in enumerate(values):
            utts = test_agents.get_round_utts(episode_idx, max_rounds)
            comp = []
            comp.append(
                [
                    "GOAL: "
                    + tod_core.SerializationHelpers.list_of_maps_to_str(
                        test_agents.make_goal_calls_machine(max_rounds)
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
        """
        Make sure goal agent data is correct with (possibly) multiple goals.
        """
        max_rounds = config[test_agents.TEST_NUM_ROUNDS_OPT_KEY]
        max_episodes = config[test_agents.TEST_NUM_EPISODES_OPT_KEY]
        values = self.dump_single_utt_per_episode_agent_text(
            test_agents.GoalAgent, config, {}
        )

        goal_text = [
            "GOAL: "
            + tod_core.SerializationHelpers.list_of_maps_to_str(
                test_agents.make_goal_calls_machine(max_rounds)
            )
            for _ in range(max_episodes)
        ]

        self.assertEqual(values, goal_text)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestApiSchemaAgent(TestTodAgentsAndTeachersBase):
    def _test_roundDataCorrect_helper(self, config):
        """
        Make sure api schema information is correct with (possibly) multiple goals.
        """
        max_rounds = config[test_agents.TEST_NUM_ROUNDS_OPT_KEY]
        max_episodes = config[test_agents.TEST_NUM_EPISODES_OPT_KEY]
        values = self.dump_single_utt_per_episode_agent_text(
            test_agents.ApiSchemaAgent, config, {}
        )

        apis_texts = [
            "APIS: "
            + tod_core.SerializationHelpers.list_of_maps_to_str(
                test_agents.make_api_schemas_machine(max_rounds)
            )
            for _ in range(max_episodes)
        ]

        self.assertEqual(values, apis_texts)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestSingleGoalAgent(TestTodAgentsAndTeachersBase):
    def _test_roundDataCorrect_helper(self, config):
        """
        Make sure single goal agent correctly splits conversations with multiple goals
        into single goals for the agent.
        """
        max_rounds = config[test_agents.TEST_NUM_ROUNDS_OPT_KEY]
        max_episodes = config[test_agents.TEST_NUM_EPISODES_OPT_KEY]
        values = self.dump_single_utt_per_episode_agent_text(
            test_agents.SingleGoalAgent, config, {}
        )

        goal_text = []
        for _ in range(max_episodes):
            goals = test_agents.make_goal_calls_machine(max_rounds)
            for x in goals:
                goal_text.append(
                    "GOAL: " + tod_core.SerializationHelpers.list_of_maps_to_str([x])
                )

        self.assertEqual(values, goal_text)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestSingleApiSchemaAgent(TestTodAgentsAndTeachersBase):
    def _test_roundDataCorrect_helper(self, config):
        """
        Make sure single api schema agent correctly splits conversations with multiple
        goals into single goals for the agent.
        """
        max_rounds = config[test_agents.TEST_NUM_ROUNDS_OPT_KEY]
        max_episodes = config[test_agents.TEST_NUM_EPISODES_OPT_KEY]
        values = self.dump_single_utt_per_episode_agent_text(
            test_agents.SingleApiSchemaAgent, config, {}
        )

        apis_text = []
        for _ in range(max_episodes):
            apis = test_agents.make_api_schemas_machine(max_rounds)
            for x in apis:
                apis_text.append(
                    "APIS: " + tod_core.SerializationHelpers.list_of_maps_to_str([x])
                )
        self.assertEqual(values, apis_text)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestSingleGoalWithSingleApiSchemaAgent(TestTodAgentsAndTeachersBase):
    """
    Make sure the SingleGoal + SingleApiSchema agents correspond.
    """

    def _test_roundDataCorrect_helper(self, config):
        goals = self.dump_single_utt_per_episode_agent_text(
            test_agents.SingleGoalAgent, config, {}
        )
        apis = self.dump_single_utt_per_episode_agent_text(
            test_agents.SingleApiSchemaAgent, config, {}
        )

        for i in range(len(goals)):
            goal = tod_core.SerializationHelpers.str_to_goals(goals[i][len("GOALS:") :])
            api = tod_core.SerializationHelpers.str_to_api_schemas(
                apis[i][len("APIS:") :]
            )
            self.assertEqual(
                goal[0].get("api_name", None), api[0].get("api_name", None)
            )

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TestLowShot(TestTodAgentsAndTeachersBase):
    FEW_SHOT_SAMPLES = [0, 1, 5, 15]
    PERCENTAGES = [0, 0.1, 0.3, 0.5]

    def setup_agent_or_teacher(self, class_type, round_opt, opt):
        full_opts = {**round_opt, **opt}
        full_opts["datatype"] = "DUMMY"
        full_opts["datafile"] = "DUMMY"
        return class_type(full_opts)

    def test_few_shot_lengths_correct(self):
        def helper(n_shot):
            values = self.dump_teacher_text(
                test_agents.SystemTeacher,
                test_agents.EPISODE_SETUP__MULTI_EPISODE_BS,
                {"episodes_randomization_seed": 0, "n_shot": n_shot},
            )
            self.assertEqual(len(values), n_shot)

        for i in self.FEW_SHOT_SAMPLES:
            helper(i)

    def _test_subsets(self, data_dumps):
        for i in range(len(data_dumps) - 1):
            small = data_dumps[i]
            larger = data_dumps[i + 1]
            for i, episode in enumerate(small):
                self.assertEqual(episode, larger[i])

    def test_few_shot_subset(self):
        """
        Make sure specifying few-shot by n-shot works correctly.
        """

        def helper(n_shot, seed):
            return self.dump_teacher_text(
                test_agents.SystemTeacher,
                test_agents.EPISODE_SETUP__MULTI_EPISODE,
                {"episodes_randomization_seed": seed, "n_shot": n_shot},
            )

        data_dumps_seed_zero = [helper(i, 0) for i in self.FEW_SHOT_SAMPLES]
        self._test_subsets(data_dumps_seed_zero)
        data_dumps_seed_three = [helper(i, 3) for i in self.FEW_SHOT_SAMPLES]
        self._test_subsets(data_dumps_seed_three)
        self.assertNotEqual(data_dumps_seed_zero[-1], data_dumps_seed_three[-1])

    def test_percent_shot_lengths_correct(self):
        """
        Make sure specifying few-shot by percentages works correctly.
        """

        def helper(percent_shot, correct):
            values = self.dump_teacher_text(
                test_agents.SystemTeacher,
                test_agents.EPISODE_SETUP__MULTI_EPISODE_BS,  # 35 episodes
                {"episodes_randomization_seed": 0, "percent_shot": percent_shot},
            )
            self.assertEqual(len(values), correct)

        helper(0, 0)
        helper(0.1, 3)
        helper(0.3, 10)

    def test_percent_shot_subset(self):
        """
        Make sure specifying few-shot by percentages works correctly.
        """

        def helper(percent_shot, seed):
            return self.dump_teacher_text(
                test_agents.SystemTeacher,
                test_agents.EPISODE_SETUP__MULTI_EPISODE_BS,  # 35 episodes
                {"episodes_randomization_seed": seed, "percent_shot": percent_shot},
            )

        data_dumps_seed_zero = [helper(i, 0) for i in self.PERCENTAGES]
        self._test_subsets(data_dumps_seed_zero)
        data_dumps_seed_three = [helper(i, 3) for i in self.PERCENTAGES]
        self._test_subsets(data_dumps_seed_three)

    def test_correct_throw_when_both_shots_defined(self):
        self.assertRaises(
            RuntimeError,
            self.dump_teacher_text,
            test_agents.SystemTeacher,
            test_agents.EPISODE_SETUP__MULTI_EPISODE_BS,  # 35 episodes
            {"episodes_randomization_seed": 0, "percent_shot": 0.3, "n_shot": 3},
        )


if __name__ == "__main__":
    unittest.main()
