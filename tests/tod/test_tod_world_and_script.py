#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests tod world + script, notably for batching, by comparing saved script logs to the
data that should have been generated.

Metrics are handled in separate files.
"""

import copy
import unittest

import parlai.core.tod.tod_test_utils.test_agents as test_agents
import parlai.core.tod.tod_core as tod_core
import parlai.scripts.tod_world_script as tod_world_script
from parlai.core.tod.tod_agents import StandaloneApiAgent


class TestTodWorldScript(tod_world_script.TodWorldScript):
    """
    Wrap around it to check its logic; also makes it easier to do things w/ underlying
    World.
    """

    def _get_tod_agents(self, opt):
        """
        Hack so we can separate out logic of making sure agent parsing is correct.
        """
        if hasattr(self, "agents"):
            return self.agents
        return super()._get_tod_agents(opt)

    def _save_outputs(self, opt, world, logger, episode_metrics):
        self.world = world
        self.logger = logger


class TodWorldInScriptTestBase(unittest.TestCase):
    def add_tod_world_opts(self, base_opts):
        """
        Convenience since we're initing the opt directly without parlai parser.
        """
        opts = copy.deepcopy(base_opts)
        opts["datatype"] = "DUMMY"
        opts["datafile"] = "DUMMY"
        opts["episodes_randomization_seed"] = -1
        opts["standalone_api_file"] = test_agents.API_DATABASE_FILE
        opts["exact_api_call"] = True
        opts["log_keep_fields"] = "all"
        opts["display_examples"] = False
        opts[
            "include_api_schemas"
        ] = True  # do this to test_agents.make sure they're done correctly.
        return opts

    def setup_agents(self, added_opts):
        full_opts = self.add_tod_world_opts(added_opts)
        sys = test_agents.ApiCallAndSysUttAgent(full_opts)
        agents = [
            test_agents.UserUttAgent(full_opts),
            sys,
            StandaloneApiAgent(full_opts),
            sys,
            test_agents.ApiSchemaAgent(full_opts),
            test_agents.GoalAgent(full_opts),
        ]
        return agents, full_opts

    def _test_roundDataCorrect(self):
        self._test_roundDataCorrect_helper(test_agents.EPISODE_SETUP__SINGLE_API_CALL)
        self._test_roundDataCorrect_helper(test_agents.EPISODE_SETUP__MULTI_ROUND)
        self._test_roundDataCorrect_helper(test_agents.EPISODE_SETUP__MULTI_EPISODE)
        self._test_roundDataCorrect_helper(test_agents.EPISODE_SETUP__MULTI_EPISODE_BS)

    def _check_correctness_from_script_logs(
        self, script, opt, process_round_utts=lambda x: x
    ):
        """
        Last argument is only relevant for the max_turn test.
        """
        max_rounds = opt[test_agents.TEST_NUM_ROUNDS_OPT_KEY]
        max_episodes = opt[test_agents.TEST_NUM_EPISODES_OPT_KEY]
        # there's something funky with logger.get_log() that inserts a space, but not gonna worry about it for now
        logs = [x for x in script.logger.get_logs() if len(x) > 0]
        for episode_idx in range(max_episodes):
            episode_from_world = logs[episode_idx]
            # first round is context
            context = episode_from_world[0]
            self.assertEquals(
                context[0]["text"],
                "APIS: "
                + tod_core.SerializationHelpers.list_of_maps_to_str(
                    test_agents.make_api_schemas_machine(max_rounds)
                ),
            )
            self.assertEquals(
                context[3]["text"],
                "GOAL: "
                + tod_core.SerializationHelpers.list_of_maps_to_str(
                    test_agents.make_goal_calls_machine(max_rounds)
                ),
            )
            # Check the rest
            world_utts = [[x["text"] for x in turn] for turn in episode_from_world[1:]]
            # ... ignore the last DONE turn here cause it's not that important

            self.assertEquals(
                world_utts[:-1],
                process_round_utts(
                    test_agents.get_round_utts(episode_idx, max_rounds)[:-1]
                ),
            )


class TodWorldSingleBatchTest(TodWorldInScriptTestBase):
    """
    Checks that saved data is correct with a single batch.
    """

    def _test_roundDataCorrect_helper(self, config):
        config["batchsize"] = 1
        config["max_turns"] = 10
        agents, opt = self.setup_agents(config)
        script = TestTodWorldScript(opt)
        script.agents = agents
        script.run()
        self._check_correctness_from_script_logs(script, opt)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()

    def test_max_turn(self):
        self._test_max_turn_helper(4)
        self._test_max_turn_helper(7)

    def _test_max_turn_helper(self, max_turns):
        config = {}
        config["batchsize"] = 1
        config["max_turns"] = max_turns
        config[test_agents.TEST_NUM_ROUNDS_OPT_KEY] = 10
        config[test_agents.TEST_NUM_EPISODES_OPT_KEY] = 5  # cause why not
        agents, opt = self.setup_agents(config)
        script = TestTodWorldScript(opt)
        script.agents = agents
        script.run()

        def filter_round_utt(utts):
            # tad imprecise, but more important that it does stop.
            # subtract 1 for the context turn, then 1 cause there's an off by one somewhere
            return utts[: max_turns - 2]

        self._check_correctness_from_script_logs(script, opt, filter_round_utt)


class TodWorldNonSingleBatchTest(TodWorldInScriptTestBase):
    """
    Checks saved data is correct with larger batchsizes.
    """

    def _test_roundDataCorrect_helper(self, config):
        config["batchsize"] = 4
        config["max_turns"] = 10
        agents, opt = self.setup_agents(config)
        script = TestTodWorldScript(opt)
        script.agents = agents
        script.run()
        self._check_correctness_from_script_logs(script, opt)

    def test_roundDataCorrect(self):
        self._test_roundDataCorrect()


class TodWorldTestSingleDumpAgents(TodWorldInScriptTestBase):
    """
    Just to be safe, make sure that the agents with "single" versions (ex goal + api
    schema) are correctly aligned.

    (Already tested in the agents test file as well, but to be safe.)
    """

    def setup_agents(self, added_opts, api_agent, goal_agent):
        full_opts = self.add_tod_world_opts(added_opts)
        full_opts["fixed_response"] = "USER: [DONE]"
        sys = test_agents.ApiCallAndSysUttAgent(full_opts)
        agents = [
            test_agents.UserUttAgent(full_opts),
            sys,
            StandaloneApiAgent(full_opts),
            sys,
            api_agent(full_opts),
            goal_agent(full_opts),
        ]
        return agents, full_opts

    def _test_SingleGoalApiResp_helper(self, batchsize, num_episodes):
        config = {}
        config["batchsize"] = batchsize
        config[test_agents.TEST_NUM_ROUNDS_OPT_KEY] = 10
        config[test_agents.TEST_NUM_EPISODES_OPT_KEY] = num_episodes
        single_agents, opt = self.setup_agents(
            config, test_agents.SingleApiSchemaAgent, test_agents.SingleGoalAgent
        )
        single_script = TestTodWorldScript(opt)
        single_script.agents = single_agents
        single_script.run()
        single_logs = [x for x in single_script.logger.get_logs() if len(x) > 0]

        multi_agents, opt = self.setup_agents(
            config, test_agents.ApiSchemaAgent, test_agents.GoalAgent
        )
        multi_script = TestTodWorldScript(opt)
        multi_script.agents = multi_agents
        multi_script.run()
        multi_logs = [x for x in single_script.logger.get_logs() if len(x) > 0]

        single_idx = 0
        for multi_log in multi_logs:
            context = multi_log[0]
            goals = tod_core.SerializationHelpers.str_to_goals(
                context[3]["text"][len("GOAL:") :].strip()
            )
            for goal in goals:
                single_context = single_logs[single_idx][0]
                single_goal = tod_core.SerializationHelpers.str_to_goals(
                    single_context[3]["text"][len("GOAL:") :].strip()
                )
                self.assertEqual(len(single_goal), 1)
                self.assertEquals(goal, single_goal[0])
                single_des = tod_core.SerializationHelpers.str_to_api_schemas(
                    single_context[0]["text"][len("APIS:") :].strip()
                )
                self.assertEqual(len(single_des), 1)
                self.assertEqual(single_goal[0]["api_name"], single_des[0]["api_name"])

                single_idx += 1

    def test_SingleGoalApiResp_helper_singleBatch(self):
        self._test_SingleGoalApiResp_helper(1, 2)
        self._test_SingleGoalApiResp_helper(1, 5)

    def test_SingleGoalApiResp_helper_multiBatch(self):
        self._test_SingleGoalApiResp_helper(4, 8)
        self._test_SingleGoalApiResp_helper(4, 11)


if __name__ == "__main__":
    unittest.main()
