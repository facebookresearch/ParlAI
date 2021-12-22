#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests tod world metrics in the full script, *including* making the script properly set
up the agents on its own.

Use a few of the API Call + goal hit metrics as the metric handlers to test proper
functionality.
"""

import copy
import unittest

from parlai.core.metrics import dict_report
from parlai.core.opt import Opt
from parlai.core.tod.tod_core import SerializationHelpers
import parlai.core.tod.tod_test_utils.test_agents as test_agents
from parlai.core.tod.world_metrics_handlers import METRICS_HANDLER_CLASSES_TEST_REGISTRY
import parlai.scripts.tod_world_script as tod_world_script

# Ignore lint on following line; want to have registered classes show up for tests
import projects.tod_simulator.world_metrics.extended_world_metrics  # noqa: F401

NUM_EPISODES = 35

TEST_SETUP = {
    "api_schema_grounding_model": "parlai.core.tod.tod_test_utils.test_agents:ApiSchemaAgent",
    "goal_grounding_model": "parlai.core.tod.tod_test_utils.test_agents:GoalAgent",
    "user_model": "parlai.core.tod.tod_test_utils.test_agents:UserUttAgent",
    "system_model": "parlai.core.tod.tod_test_utils.test_agents:ApiCallAndSysUttAgent",
    "api_resp_model": "fixed_response",
    test_agents.TEST_NUM_EPISODES_OPT_KEY: NUM_EPISODES,
}
TEST_SETUP_BROKEN_USER_SYSTEM = {
    "api_schema_grounding_model": "parlai.core.tod.tod_test_utils.test_agents:ApiSchemaAgent",
    "goal_grounding_model": "parlai.core.tod.tod_test_utils.test_agents:GoalAgent",
    "user_model": "fixed_response",
    "system_model": "fixed_response",
    "api_resp_model": "fixed_response",
    test_agents.TEST_NUM_EPISODES_OPT_KEY: NUM_EPISODES,
}

TEST_SETUP_EMPTY_APISCHEMA = copy.deepcopy(TEST_SETUP)
TEST_SETUP_EMPTY_APISCHEMA[
    "api_schema_grounding_model"
] = "parlai.core.tod.tod_agents:EmptyApiSchemaAgent"

TEST_SETUP_BROKEN_USER_SYSTEM_EMPTY_APISCHEMA = copy.deepcopy(
    TEST_SETUP_BROKEN_USER_SYSTEM
)
TEST_SETUP_BROKEN_USER_SYSTEM_EMPTY_APISCHEMA[
    "api_schema_grounding_model"
] = "parlai.core.tod.tod_agents:EmptyApiSchemaAgent"

DATATYPE = "valid"


class TestTodWorldScript(tod_world_script.TodWorldScript):
    """
    Wrap around it to check its logic; also makes it easier to do things w/ underlying
    World.
    """

    def __init__(self, opt: Opt):
        opt["datatype"] = DATATYPE
        # none of the below matter, but need to set to keep other code happy.
        opt["log_keep_fields"] = "all"
        opt["display_examples"] = False

        super().__init__(opt)

    def _setup_world(self):
        world = super()._setup_world()
        for i in range(len(world.batch_tod_world_metrics)):
            world.batch_tod_world_metrics[i].handlers = [
                x() for x in METRICS_HANDLER_CLASSES_TEST_REGISTRY
            ]
        return world

    def _save_outputs(self, opt, world, logger, episode_metrics):
        self.world = world
        self.logger = logger
        self.episode_metrics = episode_metrics


class TodMetricsInScriptTests(unittest.TestCase):
    def test_all_goals_hit_all_success(self):
        """
        For a setup where all the goals should be successfully hit, is it?
        """
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP, batchsize=1, num_episodes=1, target_all_goals_hit=1
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP, batchsize=1, num_episodes=32, target_all_goals_hit=1
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP, batchsize=32, num_episodes=8, target_all_goals_hit=1
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP, batchsize=32, num_episodes=33, target_all_goals_hit=1
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP,
            batchsize=32,
            num_episodes=-1,
            target_all_goals_hit=1,
            target_metrics_length=NUM_EPISODES,
        )

    def test_all_goals_hit_all_fail(self):
        """
        For a setup where all the goals should *not* be successfully hit, do they fail?
        """
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_BROKEN_USER_SYSTEM,
            batchsize=1,
            num_episodes=1,
            target_all_goals_hit=0,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_BROKEN_USER_SYSTEM,
            batchsize=1,
            num_episodes=32,
            target_all_goals_hit=0,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_BROKEN_USER_SYSTEM,
            batchsize=32,
            num_episodes=32,
            target_all_goals_hit=0,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_BROKEN_USER_SYSTEM,
            batchsize=32,
            num_episodes=33,
            target_all_goals_hit=0,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_BROKEN_USER_SYSTEM,
            batchsize=32,
            num_episodes=-1,
            target_all_goals_hit=0,
            target_metrics_length=NUM_EPISODES,
        )

    def test_all_goals_hit_all_success_emptySchema(self):
        """
        Check to make sure empty API schema doesn't have any impact on goal (Necessary
        cause original, more exhaustive implementation of goal success would separate
        between required + optional opts using the schema; make sure it doesn't impact
        anything broader)
        """
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_EMPTY_APISCHEMA,
            batchsize=1,
            num_episodes=1,
            target_all_goals_hit=1,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_EMPTY_APISCHEMA,
            batchsize=1,
            num_episodes=32,
            target_all_goals_hit=1,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_EMPTY_APISCHEMA,
            batchsize=32,
            num_episodes=32,
            target_all_goals_hit=1,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_EMPTY_APISCHEMA,
            batchsize=32,
            num_episodes=33,
            target_all_goals_hit=1,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_EMPTY_APISCHEMA,
            batchsize=32,
            num_episodes=-1,
            target_all_goals_hit=1,
            target_metrics_length=NUM_EPISODES,
        )

    def test_all_goals_hit_all_fail_emptySchema(self):
        """
        Make sure empty schema has no impact on goal success.

        (Necessary cause original, more exhaustive implementation of goal success would
        separate between required + optional opts using the schema; make sure it doesn't
        impact anything broader)
        """
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_BROKEN_USER_SYSTEM_EMPTY_APISCHEMA,
            batchsize=1,
            num_episodes=1,
            target_all_goals_hit=0,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_BROKEN_USER_SYSTEM_EMPTY_APISCHEMA,
            batchsize=1,
            num_episodes=32,
            target_all_goals_hit=0,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_BROKEN_USER_SYSTEM_EMPTY_APISCHEMA,
            batchsize=32,
            num_episodes=32,
            target_all_goals_hit=0,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_BROKEN_USER_SYSTEM_EMPTY_APISCHEMA,
            batchsize=32,
            num_episodes=33,
            target_all_goals_hit=0,
        )
        self._check_all_goals_hit_by_opt_and_batchsize(
            TEST_SETUP_BROKEN_USER_SYSTEM_EMPTY_APISCHEMA,
            batchsize=32,
            num_episodes=-1,
            target_all_goals_hit=0,
            target_metrics_length=NUM_EPISODES,
        )

    def _check_all_goals_hit_by_opt_and_batchsize(
        self,
        opt,
        batchsize,
        num_episodes,
        target_all_goals_hit,
        target_metrics_length=None,
    ):
        opt = copy.deepcopy(opt)
        opt["batchsize"] = batchsize
        opt["num_episodes"] = num_episodes
        report, metrics = self._run_opt_get_report(opt)
        self.assertEqual(report.get("all_goals_hit"), target_all_goals_hit)
        metrics_comp_length = num_episodes
        if target_metrics_length:
            metrics_comp_length = target_metrics_length
        self.assertEqual(len(metrics), metrics_comp_length)

    def _run_opt_get_report(self, opt):
        script = TestTodWorldScript(opt)
        script.run()

        def get_episode_report(goal, episode_metric):
            metrics_dict = dict_report(episode_metric.report())
            metrics_dict["goal"] = goal
            return metrics_dict

        return (
            dict_report(script.world.report()),
            [get_episode_report(g, e) for g, e in script.episode_metrics],
        )

    def test_apiCallAttempts_usingGold(self):
        opt = copy.deepcopy(TEST_SETUP)
        opt["batchsize"] = 1
        opt["num_episodes"] = -1
        _, metrics = self._run_opt_get_report(opt)
        for metric in metrics:
            self.assertEqual(
                len(
                    SerializationHelpers.str_to_goals(
                        metric["goal"]["text"][len("GOALS: ") :]
                    )
                ),
                metric["call_attempts"],
            )


if __name__ == "__main__":
    unittest.main()
