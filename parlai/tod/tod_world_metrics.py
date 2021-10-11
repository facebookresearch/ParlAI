#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Metrics for TODWorld.
"""

from parlai.core.message import Message
from parlai.core.metrics import (
    Metric,
    Metrics,
    AverageMetric,
    normalize_answer,
    BleuMetric,
    SumMetric,
)
from parlai.tod.tod_core import (
    STANDARD_API_NAME_SLOT,
    STANDARD_DONE,
    STANDARD_REQUIRED_KEY,
    STANDARD_OPTIONAL_KEY,
    STANDARD_API_DESCRIPTIONS,
    TodAgentType,
    TOD_AGENT_TYPE_TO_PREFIX,
    SerializationHelpers,
)
from typing import Any, Dict, List, Optional, Tuple

try:
    from nltk.translate import bleu_score as nltkbleu
except ImportError:
    # User doesn't have nltk installed, so we can't use it for bleu
    # We'll just turn off things, but we might want to warn the user
    nltkbleu = None


class TodMetricsHandler:
    """
    Base class for Tod Metrics handlers.

    Override as necessary.
    """

    def handle_api_description(self, message: Message) -> Optional[Dict[str, Metric]]:
        pass

    def handle_goal(self, message: Message) -> Optional[Dict[str, Metric]]:
        pass

    def handle_user_utt(self, message: Message) -> Optional[Dict[str, Metric]]:
        pass

    def handle_api_call(self, message: Message) -> Optional[Dict[str, Metric]]:
        pass

    def handle_api_resp(self, message: Message) -> Optional[Dict[str, Metric]]:
        pass

    def handle_sys_utt(self, message: Message) -> Optional[Dict[str, Metric]]:
        pass

    def episode_reset(self) -> Optional[Dict[str, Metric]]:
        """
        Resets entire episode, while also reporting any relevant episode-level metrics.
        """
        raise NotImplementedError("Must implement episode_reset")


class _GoalCoverageMetricHandler(TodMetricsHandler):
    """
    Some base functionality for working with machine-structured goals.
    """

    def __init__(self):
        self.goals = []
        self.required_goals = []
        self.api_descriptions = []

    def handle_api_description(self, message: Message) -> Optional[Dict[str, Metric]]:
        self.api_descriptions = SerializationHelpers.str_to_api_descriptions(
            message["prefix_stripped_text"]
        )

    def handle_goal(self, message: Message) -> Optional[Dict[str, Metric]]:
        self.goals = SerializationHelpers.str_to_goals(message["prefix_stripped_text"])
        # can do since goal is always after api descriptions
        self.required_goals = self.get_req_only_goals()

    def get_req_only_goals(self) -> Dict:
        """
        If we somehow can't parse for required goals properly, return the full goal.
        """
        if len(self.api_descriptions) == 0:
            return self.goals
        result = []
        for goal in self.goals:
            req_goals = {}
            method = goal.get(STANDARD_API_NAME_SLOT, None)
            if method is None:
                method = goal.get("api_name", None)
            if method is None or method == "":
                return self.goals
            required = []
            for description in self.api_descriptions:
                if description.get(STANDARD_API_NAME_SLOT, "") == method:
                    required = description.get(STANDARD_REQUIRED_KEY, {})
            for key in required:
                if key not in goal:
                    return self.goals
                req_goals[key] = goal[key]
            if len(req_goals) > 0:
                req_goals[STANDARD_API_NAME_SLOT] = method  # for consistency with all.
                result.append(req_goals)
        return result


class ApiCallHitGoalMetricHandler(_GoalCoverageMetricHandler):
    """
    Given a machine-formatted goal passed to the user agent, does the system agent
    eventually make an API call that matches this goal?

    How quickly does this occur?
    """

    def __init__(self):
        super().__init__()
        self.api_turns = []

    def handle_api_call(self, message: Message) -> Optional[Dict[str, Metric]]:
        call = SerializationHelpers.str_to_api_dict(message["prefix_stripped_text"])
        if len(call) > 0:
            self.api_turns.append(call)

    def episode_reset(self) -> Optional[Dict[str, Metric]]:
        if len(self.goals) == 0 and len(self.api_turns) == 0:
            self.goals = []
            self.api_turns = []
            self.api_descriptions = []
            return {}
        all_goals_hit, all_goals_hit_turn_count, all_part_hit = self.goals_hit_helper(
            self.goals, self.api_turns
        )
        all_goals_slot_precision, all_goals_slot_recall = self.goal_slots_helper(
            self.goals, self.api_turns
        )
        req_goals_hit, req_goals_hit_turn_count, req_part_hit = self.goals_hit_helper(
            self.required_goals, self.api_turns
        )
        req_goals_slot_precision, req_goals_slot_recall = self.goal_slots_helper(
            self.required_goals, self.api_turns
        )
        call_attempts = len(self.api_turns)
        self.goals = []
        self.api_turns = []
        self.api_descriptions = []
        return {
            "all_goals_hit": all_goals_hit,
            "all_goals_hit_turn_count": all_goals_hit_turn_count,
            "all_goals_fractional_hit": all_part_hit,
            "all_goals_slot_precision": all_goals_slot_precision,
            "all_goals_slot_recall": all_goals_slot_recall,
            "req_goals_hit": req_goals_hit,
            "req_goals_hit_turn_count": req_goals_hit_turn_count,
            "req_goals_fractional_hit": req_part_hit,
            "req_goals_slot_precision": req_goals_slot_precision,
            "req_goals_slot_recall": req_goals_slot_recall,
            "call_attempts": AverageMetric(call_attempts),
        }

    def goal_slots_helper(
        self, goals: List[Dict], turnDict: List[Dict]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Output is precision, recall.
        """
        all_call_slots = {k: v for call in turnDict for k, v in call.items()}
        all_goal_slots = {k: v for goal in goals for k, v in goal.items()}
        goal_in_call = {
            k: v
            for k, v in all_call_slots.items()
            if all_goal_slots.get(k, "definitelyNotInValuexyz") == v
        }
        call_in_goal = {
            k: v
            for k, v in all_goal_slots.items()
            if all_call_slots.get(k, "definitelyNotInValuexyz") == v
        }

        return (
            AverageMetric(len(goal_in_call), len(all_call_slots)),
            AverageMetric(len(call_in_goal), len(all_goal_slots)),
        )

    def goals_hit_helper(
        self, goals: List[Dict], turnDict: List[Dict]
    ) -> (AverageMetric, AverageMetric, AverageMetric):
        """
        Return values:

        * if all goals hit
        * # of turns it took to hit all goals (or None)
        * fraction of goals hit
        """
        goals_left = goals
        for i, turn in enumerate(turnDict):
            goals_left = [
                goal for goal in goals_left if not self.goal_hit_helper(goal, turn)
            ]
            if len(goals_left) == 0:
                return AverageMetric(True), AverageMetric(i + 1), AverageMetric(1)
        return (
            AverageMetric(False),
            AverageMetric(0),
            AverageMetric(len(goals) - len(goals_left), len(goals)),
        )

    def goal_hit_helper(self, goal: Dict, turn: Dict) -> bool:
        for key in goal:
            if key not in turn:
                return False
            if goal[key] != turn[key]:
                return False
        return True


class UserGoalSlotCoverageMetricHandler(_GoalCoverageMetricHandler):
    """
    How well does our user simulator do at outputting utterances that goes closer to
    satisfying relevant preempted goals? Does it dump out all of the slots at once or is
    it more intelligent than that?

    Since this is the user and we don't know the identity of potential slots, we ignore
    the short (< 4 chars) goal slots since this tends to be things that are substrings
    of other things. (Ex. "2" showing up as # of people in a reservation, but also
    showing up as a phone number.)
    """

    def __init__(self):
        super().__init__()
        self.user_utt_all_slots = set()
        self.user_utt_req_slots = set()
        self.all_goal_entities = set()
        self.all_req_goal_entities = set()

    def handle_goal(self, message: Message) -> Optional[Dict[str, Metric]]:
        """
        Parse out all the slots as a blob, filtering out for short things.
        """
        super().handle_goal(message)

        def get_entities(goal_list):
            result = set()
            for goal in goal_list:
                for key, value in goal.items():
                    if key is not STANDARD_API_NAME_SLOT and len(value) > 3:
                        result.add(value)
            return result

        self.all_goal_entities = get_entities(self.goals)
        self.all_req_goal_entities = get_entities(self.required_goals)

    def handle_user_utt(self, message: Message) -> Optional[Dict[str, Metric]]:
        """
        Grab slots out of the user utterance based on an exact match.
        """
        utterance = message["prefix_stripped_text"]

        def get_slots(utt, options):
            results = set()
            for option in options:
                if option in utt:
                    results.add(option)
            return results

        all_entities_here = get_slots(utterance, self.all_goal_entities)
        req_entities_here = get_slots(utterance, self.all_req_goal_entities)

        self.user_utt_all_slots |= all_entities_here
        self.user_utt_req_slots |= req_entities_here

        metrics = {}
        metrics["user_utt_avg_any_slot"] = AverageMetric(len(all_entities_here))
        metrics["user_utt_avg_req_slot"] = AverageMetric(len(req_entities_here))
        return metrics

    def episode_reset(self) -> Optional[Dict[str, Metric]]:
        result = {
            "user_any_goal_slots_recall": AverageMetric(
                len(self.user_utt_all_slots), len(self.all_goal_entities)
            ),
            "user_req_goal_slots_recall": AverageMetric(
                len(self.user_utt_req_slots), len(self.all_req_goal_entities)
            ),
        }

        self.user_utt_all_slots = set()
        self.user_utt_req_slots = set()
        return result


class _ExactRepeatMetricsHandler(TodMetricsHandler):
    """
    % of episodes where a given agent type has exactly repeated the same utterance.
    """

    def __init__(self):
        self.turns = []
        self.repeated = False

    def metric_key(self):
        raise NotImplementedError("must implement")

    def handle_message_helper(self, message: Message) -> Optional[Dict[str, Metric]]:
        normalized = normalize_answer(message["prefix_stripped_text"])
        if normalized in self.turns:
            self.repeated = True
        self.turns.append(normalized)

    def episode_reset(self) -> Optional[Dict[str, Metric]]:
        repeat = int(self.repeated)
        self.repeated = False
        self.turns = []
        return {self.metric_key(): AverageMetric(repeat)}


class UserUttRepeatMetricHandler(_ExactRepeatMetricsHandler):
    def metric_key(self):
        return "user_utt_repeat"

    def handle_user_utt(self, message: Message) -> Optional[Dict[str, Metric]]:
        return self.handle_message_helper(message)


class SystemUttRepeatMetricHandler(_ExactRepeatMetricsHandler):
    def metric_key(self):
        return "sys_utt_repeat"

    def handle_sys_utt(self, message: Message) -> Optional[Dict[str, Metric]]:
        return self.handle_message_helper(message)


class _Bleu3MetricsHandler(TodMetricsHandler):
    """
    For a given agent, this calculates the Bleu-3 of a new turn against prior turns.

    This is an alternate metric for repetativeness
    """

    def __init__(self):
        self.turns = []

    def metric_key(self):
        raise NotImplementedError("must implement")

    def handle_message_helper(self, message: Message) -> Optional[Dict[str, Metric]]:
        here = normalize_answer(message["prefix_stripped_text"]).split(" ")
        score = 1
        if len(self.turns) > 0:
            score = nltkbleu.corpus_bleu(
                [self.turns],
                [here],
                smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
                weights=[1.0 / 3.0] * 3,
            )
        self.turns.append(here)
        return {self.metric_key(): BleuMetric(score)}

    def episode_reset(self) -> Optional[Dict[str, Metric]]:
        self.turns = []


class UserUttSelfBleu3MetricHandler(_Bleu3MetricsHandler):
    def metric_key(self):
        return "user_utt_self_bleu3"

    def handle_user_utt(self, message: Message) -> Optional[Dict[str, Metric]]:
        return self.handle_message_helper(message)


class SystemUttSelfBleu3MetricHandler(_Bleu3MetricsHandler):
    def metric_key(self):
        return "sys_utt_self_bleu3"

    def handle_sys_utt(self, message: Message) -> Optional[Dict[str, Metric]]:
        return self.handle_message_helper(message)


class ApiCallMalformedMetricHandler(TodMetricsHandler):
    def __init__(self):
        self.api_descriptions = []

    def handle_api_description(self, message: Message) -> Optional[Dict[str, Metric]]:
        self.api_descriptions = SerializationHelpers.str_to_api_descriptions(
            message["prefix_stripped_text"]
        )

    def handle_api_call(self, message: Message) -> Optional[Dict[str, Metric]]:
        if STANDARD_API_DESCRIPTIONS in message["text"]:
            return  # Happens for API call preemption
        parsed = SerializationHelpers.str_to_api_dict(message["prefix_stripped_text"])
        if len(parsed) == 0:
            return
        if STANDARD_API_NAME_SLOT not in parsed:
            return {
                "apiCall_wellFormed": AverageMetric(0),
                "apiCall_hasSlotsButNoApiNameSlot_count": SumMetric(1),
            }
        method = parsed[STANDARD_API_NAME_SLOT]

        method_found = False
        if len(self.api_descriptions) > 0:
            for description in self.api_descriptions:
                if method == description.get(STANDARD_API_NAME_SLOT, ""):
                    method_found = True
                    check = parsed.keys()
                    required = set(description.get(STANDARD_REQUIRED_KEY, []))
                    required.add(STANDARD_API_NAME_SLOT)
                    for req in required:
                        if req not in check:  # miissing required
                            return {
                                "apiCall_wellFormed": AverageMetric(0),
                                "apiCall_missingRequiredSlot_count": SumMetric(1),
                            }
                    opt_count = 0
                    for opt in description.get(STANDARD_OPTIONAL_KEY, []):
                        if opt in check:
                            opt_count += 1
                    if opt_count + len(required) != len(check):
                        # have extra APIs that are not
                        return {
                            "apiCall_wellFormed": AverageMetric(0),
                            "apiCall_hasExtraParams_count": SumMetric(1),
                        }
                    break
        if method_found:
            return {
                "apiCall_wellFormed": AverageMetric(1),
                "apiCall_wellFormed_count": SumMetric(1),
            }
        return {
            "apiCall_wellFormed": AverageMetric(0),
            "apiCall_methodDNE_count": SumMetric(1),
        }

    def episode_reset(self) -> Optional[Dict[str, Metric]]:
        self.api_descriptions = []
        return {}


class UserGeneratedDoneMetricHandler(TodMetricsHandler):
    def __init__(self):
        self.done_seen = False
        self.turn_count = 0

    def handle_user_utt(self, message: Message) -> Optional[Dict[str, Metric]]:
        self.done_seen |= STANDARD_DONE in message["text"]
        self.turn_count += 1

    def episode_reset(self) -> Optional[Dict[str, Metric]]:
        result = {"done_seen": AverageMetric(self.done_seen)}
        if self.done_seen:
            result["user_done_seen_convo_length"] = AverageMetric(self.turn_count)
        result["rounds_count"] = AverageMetric(self.turn_count)
        self.done_seen = False
        self.turn_count = 0
        return result


class PseudoInformMetricsHandler(TodMetricsHandler):
    """
    Pseudo-inform rate.
    """

    def __init__(self):
        self.api_resp_slots = {}

    def handle_api_resp(self, message: Message) -> Optional[Dict[str, Metric]]:
        self.api_resp_slots.update(
            SerializationHelpers.str_to_api_dict(message["prefix_stripped_text"])
        )

    def handle_sys_utt(self, message: Message) -> Optional[Dict[str, Metric]]:
        m = message["prefix_stripped_text"]
        count = 0
        for val in self.api_resp_slots.values():
            if val in m:
                count += 1
        result = {"pseudo_inform_allSysTurns": AverageMetric(count)}
        if len(self.api_resp_slots) > 0:
            result["pseudo_inform_postApiRespSysTurns"] = AverageMetric(count)
        return result

    def episode_reset(self):
        self.api_resp_slots = {}


class TodMetrics(Metrics):
    """
    Helper container which encapsulates TOD metrics.
    """

    TOD_AGENT_TYPE_TO_HANDLER = {
        TodAgentType.API_DESCRIPTION_PREEMPT_AGENT: "handle_api_description",
        TodAgentType.GOAL_PREEMPT_AGENT: "handle_goal",
        TodAgentType.USER_UTT_AGENT: "handle_user_utt",
        TodAgentType.API_CALL_AGENT: "handle_api_call",
        TodAgentType.API_RESP_AGENT: "handle_api_resp",
        TodAgentType.SYSTEM_UTT_AGENT: "handle_sys_utt",
    }

    def __init__(self, shared: Dict[str, Any] = None) -> None:
        super().__init__(shared=shared)
        self.handlers = [
            ApiCallHitGoalMetricHandler(),
            UserGoalSlotCoverageMetricHandler(),
            UserUttRepeatMetricHandler(),
            SystemUttRepeatMetricHandler(),
            ApiCallMalformedMetricHandler(),
            UserUttSelfBleu3MetricHandler(),
            SystemUttSelfBleu3MetricHandler(),
            UserGeneratedDoneMetricHandler(),
            PseudoInformMetricsHandler(),
        ]
        self.convo_started = False

    def handle_message(self, message: Message, agent_type: TodAgentType):
        if "text" not in message:
            return
        if agent_type == TodAgentType.GOAL_PREEMPT_AGENT:
            # Only count a conversation as started if there is a goal.
            self.convo_started = True
        # as a convenience, do a prefix stripping and add to message
        if "prefix_stripped_text" not in message:
            message["prefix_stripped_text"] = message["text"].replace(
                TOD_AGENT_TYPE_TO_PREFIX[agent_type], ""
            )
        handle_func = self.TOD_AGENT_TYPE_TO_HANDLER[agent_type]
        for handler in self.handlers:
            metrics = getattr(handler, handle_func)(message)
            if metrics is not None:
                for name, metric in metrics.items():
                    if metric is not None:
                        self.add(name, metric)

    def episode_reset(self):
        if self.convo_started:
            episode_metrics = Metrics()
            for handler in self.handlers:
                metrics = handler.episode_reset()
                if metrics is not None:
                    for name, metric in metrics.items():
                        if metric is not None:
                            self.add(name, metric)
                            episode_metrics.add(name, metric)
            self.convo_started = False
            return episode_metrics
        return None
