#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Metrics handlers - ie, classes that handle generations from Tod World and calculates metrics from them.

Note that only metrics handler classes in `WORLD_METRIC_HANDLERS` of ` parlai/core/tod/world_metrics_handlers.py` are actively being recorded as metrics. These "extended" metrics were ones we experimented with at one point in the past, found to be inconclusive, and are including primarily to not delete already done work. (Or for testing purposes.)
"""

from parlai.core.message import Message
from parlai.core.metrics import (
    Metric,
    AverageMetric,
    normalize_answer,
    BleuMetric,
    SumMetric,
)
from parlai.core.tod.tod_core import (
    STANDARD_API_NAME_SLOT,
    STANDARD_REQUIRED_KEY,
    STANDARD_OPTIONAL_KEY,
    STANDARD_API_SCHEMAS,
)
from typing import Dict, List, Optional, Tuple
from parlai.core.tod.world_metrics_handlers import (
    TodMetricsHandler,
    register_metrics_handler,
    _ApiCallGoalInteractionHelper,
    goals_hit_helper,
)

try:
    from nltk.translate import bleu_score as nltkbleu
except ImportError:
    # User doesn't have nltk installed, so we can't use it for bleu
    # We'll just turn off things, but we might want to warn the user
    nltkbleu = None

################################
# Functions and classes associated with calculating statistics between API Calls and Goals.


def get_req_only_goals(goals_list: List[Dict], api_schemas: List[Dict]) -> List[Dict]:
    """
    Given a list of goals and a list of api schemas that say if slots are required or
    optional, this function filters for the goals to be only the required ones.

    If we have no api schemas or a goal is malformed, we return the empty list. If a
    goal is malformed, we print a warning, since this whole req-only goals thing is
    experimental at best anyhow.
    """
    if len(api_schemas) == 0:
        return []
    result = []
    for goal in goals_list:
        req_goals = {}
        method = goal.get(STANDARD_API_NAME_SLOT, None)
        if method is None:
            return []
        required = []
        for schema in api_schemas:
            if schema.get(STANDARD_API_NAME_SLOT, "") == method:
                required = schema.get(STANDARD_REQUIRED_KEY, {})
        print("-".join(required))
        for key in required:
            if key not in goal:
                print(f"No required key `{key}` in goal `{goal}`")
                return []
            req_goals[key] = goal[key]
        if len(req_goals) > 0:
            req_goals[STANDARD_API_NAME_SLOT] = method  # for consistency with all.
            result.append(req_goals)
    return result


def goals_slots_helper(
    goals: List[Dict], turnDict: List[Dict]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Helper function to see how well the slot keys + slot values match between attempted
    API calls and goals.

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

    print(goal_in_call, all_call_slots)

    return (
        AverageMetric(len(goal_in_call), len(all_call_slots)),
        AverageMetric(len(call_in_goal), len(all_goal_slots)),
    )


@register_metrics_handler
class LegacyGoalApiCallInteractionsMetricsHandler(_ApiCallGoalInteractionHelper):
    """
    This class was reporting a few too many metrics, but is useful for test purposes, so
    we're keeping it around.

    `AllGoalApiCallSuccessMetricsHandler` is the streamlined, less spammy version of
    this class.
    """

    def handle_goals(
        self, message: Message, goals: List[Dict]
    ) -> Optional[Dict[str, Metric]]:
        self.goals = goals
        self.required_goals = get_req_only_goals(goals, self.api_schemas)

    def get_episode_metrics(self) -> Optional[Dict[str, Metric]]:
        all_goals_hit, all_goals_hit_turn_count, all_part_hit = goals_hit_helper(
            self.goals, self.api_turns
        )
        all_precision, all_recall = goals_slots_helper(self.goals, self.api_turns)
        req_goals_hit, req_goals_hit_turn_count, req_part_hit = goals_hit_helper(
            self.required_goals, self.api_turns, permissive=True
        )
        req_precision, req_recall = goals_slots_helper(
            self.required_goals, self.api_turns
        )
        call_attempts = len(self.api_turns)
        return {
            "all_goals_hit": all_goals_hit,
            "all_goals_hit_turn_count": all_goals_hit_turn_count,
            "all_goals_fractional_hit": all_part_hit,
            "all_goals_slot_precision": all_precision,
            "all_goals_slot_recall": all_recall,
            "req_goals_hit": req_goals_hit,
            "req_goals_hit_turn_count": req_goals_hit_turn_count,
            "req_goals_fractional_hit": req_part_hit,
            "req_goals_slot_precision": req_precision,
            "req_goals_slot_recall": req_recall,
            "call_attempts": AverageMetric(call_attempts),
        }


@register_metrics_handler
class UserGoalSlotCoverageMetricHandler(TodMetricsHandler):
    """
    How well does our user simulator do at outputting utterances that goes closer to
    satisfying relevant groundinged goals? Does it dump out all of the slots at once or
    is it more intelligent than that?

    Since this is the user and we don't know the identity of potential slots, we ignore
    the short (< 4 chars) goal slots since this tends to be things that are substrings
    of other things. (Ex. "2" showing up as # of people in a reservation, but also
    showing up as a phone number.)
    """

    def episode_reset(self):
        self.mentioned_all_slot_values = set()
        self.mentioned_req_slot_values = set()
        self.all_goal_slot_values = set()
        self.all_req_goal_slot_values = set()

    def handle_goals(
        self, message: Message, goals: List[Dict]
    ) -> Optional[Dict[str, Metric]]:
        """
        Parse out all the slots as a blob, filtering out for short things.
        """
        required_goals = get_req_only_goals(goals, self.api_schemas)

        def get_slot_values(goal_list):
            result = set()
            for goal in goal_list:
                for key, value in goal.items():
                    if key is not STANDARD_API_NAME_SLOT and len(value) > 3:
                        result.add(value)
            return result

        self.all_goal_slot_values = get_slot_values(goals)
        self.all_req_goal_slot_values = get_slot_values(required_goals)

    def handle_user_utt(
        self, message: Message, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        """
        Grab slots out of the user utterance based on an exact match.
        """
        utterance = prefix_stripped_text

        def get_slots(utt, options):
            results = set()
            for option in options:
                if option in utt:
                    results.add(option)
            return results

        all_slot_values_here = get_slots(utterance, self.all_goal_slot_values)
        req_slot_values_here = get_slots(utterance, self.all_req_goal_slot_values)

        self.mentioned_all_slot_values |= all_slot_values_here
        self.mentioned_req_slot_values |= req_slot_values_here

        metrics = {}
        metrics["user_utt_avg_any_slot"] = AverageMetric(len(all_slot_values_here))
        metrics["user_utt_avg_req_slot"] = AverageMetric(len(req_slot_values_here))
        return metrics

    def get_episode_metrics(self) -> Optional[Dict[str, Metric]]:
        result = {
            "user_any_goal_slots_recall": AverageMetric(
                len(self.mentioned_all_slot_values), len(self.all_goal_slot_values)
            ),
            "user_req_goal_slots_recall": AverageMetric(
                len(self.mentioned_req_slot_values), len(self.all_req_goal_slot_values)
            ),
        }

        self.mentioned_all_slot_values = set()
        self.mentioned_req_slot_values = set()
        return result


class _ExactRepeatMetricsHandler(TodMetricsHandler):
    """
    Helper class for defining % of episodes where a given agent type has exactly
    repeated the same utterance.
    """

    def episode_reset(self):
        self.turns = []
        self.repeated = False

    def metric_key(self):
        raise NotImplementedError("must implement")

    def handle_message_helper(
        self, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        normalized = normalize_answer(prefix_stripped_text)
        if normalized in self.turns:
            self.repeated = True
        self.turns.append(normalized)

    def get_episode_metrics(self) -> Optional[Dict[str, Metric]]:
        repeat = int(self.repeated)
        self.repeated = False
        self.turns = []
        return {self.metric_key(): AverageMetric(repeat)}


@register_metrics_handler
class UserUttRepeatMetricHandler(_ExactRepeatMetricsHandler):
    def metric_key(self):
        return "user_utt_repeat"

    def handle_user_utt(
        self, message: Message, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        return self.handle_message_helper(prefix_stripped_text)


@register_metrics_handler
class SystemUttRepeatMetricHandler(_ExactRepeatMetricsHandler):
    def metric_key(self):
        return "sys_utt_repeat"

    def handle_sys_utt(
        self, message: Message, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        return self.handle_message_helper(prefix_stripped_text)


@register_metrics_handler
class _Bleu3MetricsHandler(TodMetricsHandler):
    """
    For a given agent, this calculates the Bleu-3 of a new turn against prior turns.

    This is an alternate metric for repetativeness
    """

    def episode_reset(self):
        self.turns = []

    def metric_key(self):
        raise NotImplementedError("must implement")

    def handle_message_helper(
        self, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        here = [normalize_answer(x) for x in prefix_stripped_text.split(" ")]
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


@register_metrics_handler
class UserUttSelfBleu3MetricHandler(_Bleu3MetricsHandler):
    def metric_key(self):
        return "user_utt_self_bleu3"

    def handle_user_utt(
        self, message: Message, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        return self.handle_message_helper(prefix_stripped_text)


@register_metrics_handler
class SystemUttSelfBleu3MetricHandler(_Bleu3MetricsHandler):
    def metric_key(self):
        return "sys_utt_self_bleu3"

    def handle_sys_utt(
        self, message: Message, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        return self.handle_message_helper(prefix_stripped_text)


@register_metrics_handler
class ApiCallMalformedMetricHandler(TodMetricsHandler):
    def episode_reset(self):
        self.api_schemas = []

    def handle_api_call(
        self, message: Message, api_call: Dict
    ) -> Optional[Dict[str, Metric]]:
        if STANDARD_API_SCHEMAS in message["text"]:
            return  # Happens for API call groundingion, so it's fine
        if len(api_call) == 0:
            return
        if STANDARD_API_NAME_SLOT not in api_call:
            return {
                "apiCall_wellFormed": AverageMetric(0),
                "apiCall_hasSlotsButNoApiNameSlot_count": SumMetric(1),
            }
        method = api_call[STANDARD_API_NAME_SLOT]

        method_found = False
        if len(self.api_schemas) > 0:
            for schema in self.api_schemas:
                if method == schema.get(STANDARD_API_NAME_SLOT, ""):
                    method_found = True
                    check = api_call.keys()
                    required = set(schema.get(STANDARD_REQUIRED_KEY, []))
                    required.add(STANDARD_API_NAME_SLOT)
                    for req in required:
                        if req not in check:  # miissing required
                            return {
                                "apiCall_wellFormed": AverageMetric(0),
                                "apiCall_missingRequiredSlot_count": SumMetric(1),
                            }
                    opt_count = 0
                    for opt in schema.get(STANDARD_OPTIONAL_KEY, []):
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


@register_metrics_handler
class PseudoInformMetricsHandler(TodMetricsHandler):
    """
    Pseudo-inform rate.
    """

    def episode_reset(self):
        self.api_resp_slots = {}

    def handle_api_resp(
        self, message: Message, api_resp: Dict
    ) -> Optional[Dict[str, Metric]]:
        self.api_resp_slots.update(api_resp)

    def handle_sys_utt(
        self, message: Message, prefix_stripped_text: str
    ) -> Optional[Dict[str, Metric]]:
        count = 0
        for val in self.api_resp_slots.values():
            if val in prefix_stripped_text:
                count += 1
        result = {"pseudo_inform_allSysTurns": AverageMetric(count)}
        if len(self.api_resp_slots) > 0:
            result["pseudo_inform_postApiRespSysTurns"] = AverageMetric(count)
        return result
