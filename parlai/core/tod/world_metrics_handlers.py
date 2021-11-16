#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Metrics handlers - ie, objects that handle generations from Tod World and calculates metrics from them.

Note that only metrics handler classes in `WORLD_METRIC_HANDLERS` (of `world_metrics.py`) are actively being recorded as metrics.
"""

from parlai.core.message import Message
from parlai.core.metrics import (
    Metric,
    AverageMetric,
)
from parlai.core.tod.tod_core import (
    STANDARD_DONE,
)
from typing import Dict, List, Optional

METRICS_HANDLER_CLASSES_TEST_REGISTRY = set()  # for tests


def register_metrics_handler(cls):
    METRICS_HANDLER_CLASSES_TEST_REGISTRY.add(cls)
    return cls


class TodMetricsHandler:
    """
    Base for Tod Metrics handlers class. The `TodMetrics` class will call the following.

    Override as necessary.
    """

    def __init__(self):
        self.episode_reset()

    def episode_reset(self):
        pass

    def handle_api_schemas(
        self, message: Message, api_schemas: List[Dict]
    ) -> Optional[Dict[str, Metric]]:
        self.api_schemas = api_schemas

    def handle_goals(
        self, message: Message, goals: List[Dict]
    ) -> Optional[Dict[str, Metric]]:
        self.goals = goals

    def handle_user_utt(
        self, message: Message, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        pass

    def handle_api_call(
        self, message: Message, api_call: Dict
    ) -> Optional[Dict[str, Metric]]:
        pass

    def handle_api_resp(
        self, message: Message, api_resp: Dict
    ) -> Optional[Dict[str, Metric]]:
        pass

    def handle_sys_utt(
        self, message: Message, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        pass

    def get_episode_metrics(self) -> Optional[Dict[str, Metric]]:
        pass


################################
# Functions and classes associated with calculating statistics between API Calls and Goals.
def goals_hit_helper(
    goals: List[Dict], turnDict: List[Dict], permissive=False
) -> (AverageMetric, AverageMetric, AverageMetric):
    """
    Helper function that aids in seeing if the API calls the system has attempted to
    make manages to meet the goals the conversation has.

    Return values:
    * if all goals hit
    * # of turns it took to hit all goals (or None)
    * fraction of goals hit
    """
    goals_left = goals

    def exact_match(goal, turn):  # if and only if
        return goal == turn

    def permissive_match(goal, turn):  # guess is superset
        for key in goal:
            if turn.get(key, "definitelyNotIn") != goal[key]:
                return False
        return True

    compare_func = permissive_match if permissive else exact_match

    for i, turn in enumerate(turnDict):
        goals_left = [goal for goal in goals_left if not compare_func(goal, turn)]
        if len(goals_left) == 0:
            return AverageMetric(True), AverageMetric(i + 1), AverageMetric(1)
    return (
        AverageMetric(False),
        AverageMetric(0),
        AverageMetric(len(goals) - len(goals_left), len(goals)),
    )


class _ApiCallGoalInteractionHelper(TodMetricsHandler):
    """
    Base class for storing details about valid API calls (information about Goals
    handled in TodMetricsHandler)
    """

    def episode_reset(self):
        self.api_turns = []

    def handle_api_call(
        self, message: Message, api_call: Dict
    ) -> Optional[Dict[str, Metric]]:
        if len(api_call) > 0:
            self.api_turns.append(api_call)


@register_metrics_handler
class AllGoalApiCallSuccessMetricsHandler(_ApiCallGoalInteractionHelper):
    """
    Calculates synthetic Task Success + related metrics for converseations.

    Test coverage of this class is with `LegacyGoalApiCallInteractionsMetricsHandler`
    """

    def get_episode_metrics(self) -> Optional[Dict[str, Metric]]:
        all_goals_hit, _, _ = goals_hit_helper(self.goals, self.api_turns)
        call_attempts = len(self.api_turns)
        return {
            "synthetic_task_success": all_goals_hit,
            "api_call_attempts": AverageMetric(call_attempts),
        }


@register_metrics_handler
class UserGeneratedDoneMetricHandler(TodMetricsHandler):
    def episode_reset(self):
        self.done_seen = False
        self.turn_count = 0

    def handle_user_utt(
        self, message: Message, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        self.done_seen |= STANDARD_DONE in message["text"]
        self.turn_count += 1

    def get_episode_metrics(self) -> Optional[Dict[str, Metric]]:
        result = {"done_seen": AverageMetric(self.done_seen)}
        if self.done_seen:
            result["round_count_done_seen"] = AverageMetric(self.turn_count)
        result["rounds_count_all_conversations"] = AverageMetric(self.turn_count)
        return result
