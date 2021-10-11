#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests tod world, notably for batching.
"""

import copy
import unittest

from parlai import __file__ as PARLAI_FILE
import parlai.tod.tod_test_utils.agents_and_teachers as aat
import parlai.tod.tod_core as tod_core
import parlai.tod.scripts.tod_world_script as tod_world_script
from parlai.tod.tod_agents import TodStandaloneApiAgent

import os


class TestTodWorldScript(tod_world_script.TodWorldScript):
    """
    Wrap around it to check its logic; also aat.makes it easier to do things w/
    underlying World.
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
        self.episode_metrics = episode_metrics


class TodWorldInScriptTestBase(unittest.TestCase):
    def add_tod_world_opts(self, base_opts):
        """
        Convenience since we're initing the opt directly without parlai parser.
        """
        opts = copy.deepcopy(base_opts)
        opts["datatype"] = "DUMMY"
        opts["datafile"] = "DUMMY"
        opts["episodes_randomization_seed"] = 32  # test it!
        opts["standalone_api_file"] = os.path.join(
            os.path.dirname(PARLAI_FILE),
            "tod/tod_test_utils/standalone_api_file.pickle",
        )  # hope this doesn't break anything...
        opts["exact_api_call"] = True
        opts["log_keep_fields"] = "all"
        opts["display_examples"] = False
        opts[
            "include_api_descriptions"
        ] = True  # do this to aat.make sure they're done correctly.
        return opts

    def setup_agents(self, added_opts):
        full_opts = self.add_tod_world_opts(added_opts)
        sys = aat.ApiCallAndSysUttAgent(full_opts)
        agents = [
            aat.UserUttAgent(full_opts),
            sys,
            TodStandaloneApiAgent(full_opts),
            sys,
            aat.ApiDescriptionAgent(full_opts),
            aat.GoalAgent(full_opts),
        ]
        return agents, full_opts

    def _test_roundDataCorrect(self):
        self._test_roundDataCorrect_helper(aat.EPISODE_SETUP__SINGLE_API_CALL)
        self._test_roundDataCorrect_helper(aat.EPISODE_SETUP__MULTI_ROUND)
        self._test_roundDataCorrect_helper(aat.EPISODE_SETUP__MULTI_EPISODE)
        self._test_roundDataCorrect_helper(aat.EPISODE_SETUP__MULTI_EPISODE_BS)

    def _check_correctness_from_script_logs(
        self, script, opt, process_round_utts=lambda x: x
    ):
        """
        Last argument is only relevant for the max_turn test.
        """
        max_rounds = opt[aat.TEST_NUM_ROUNDS_OPT_KEY]
        max_episodes = opt[aat.TEST_NUM_EPISODES_OPT_KEY]
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
                    aat.make_api_descriptions_machine(max_rounds)
                ),
            )
            self.assertEquals(
                context[3]["text"],
                "GOAL: "
                + tod_core.SerializationHelpers.list_of_maps_to_str(
                    aat.make_goal_calls_machine(max_rounds)
                ),
            )
            # Check the rest
            world_utts = [[x["text"] for x in turn] for turn in episode_from_world[1:]]
            # ... ignore the last DONE turn here cause it's not that important
            print(world_utts)

            self.assertEquals(
                world_utts[:-1],
                process_round_utts(aat.get_round_utts(episode_idx, max_rounds)[:-1]),
            )


class TodWorldSingleBatchTest(TodWorldInScriptTestBase):
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
        config[aat.TEST_NUM_ROUNDS_OPT_KEY] = 10
        config[aat.TEST_NUM_EPISODES_OPT_KEY] = 2  # cause why not
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
    def setup_agents(self, added_opts, api_agent, goal_agent):
        full_opts = self.add_tod_world_opts(added_opts)
        full_opts["fixed_response"] = "USER: [DONE]"
        sys = aat.ApiCallAndSysUttAgent(full_opts)
        agents = [
            aat.UserUttAgent(full_opts),
            sys,
            TodStandaloneApiAgent(full_opts),
            sys,
            api_agent(full_opts),
            goal_agent(full_opts),
        ]
        return agents, full_opts

    def test_SingleGoalApiResp_noBatching(self):
        config = {}
        config["batchsize"] = 1
        config[aat.TEST_NUM_ROUNDS_OPT_KEY] = 10
        config[aat.TEST_NUM_EPISODES_OPT_KEY] = 2  # cause why not
        single_agents, opt = self.setup_agents(
            config, aat.SingleApiDescriptionAgent, aat.SingleGoalAgent
        )
        single_script = TestTodWorldScript(opt)
        single_script.agents = single_agents
        single_script.run()
        single_logs = [x for x in single_script.logger.get_logs() if len(x) > 0]

        multi_agents, opt = self.setup_agents(
            config, aat.ApiDescriptionAgent, aat.GoalAgent
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
                single_des = tod_core.SerializationHelpers.str_to_api_descriptions(
                    single_context[0]["text"][len("APIS:") :].strip()
                )
                self.assertEqual(len(single_des), 1)
                print(single_goal, single_des)
                self.assertEqual(single_goal[0]["api_name"], single_des[0]["api_name"])

                single_idx += 1


if __name__ == "__main__":
    unittest.main()
