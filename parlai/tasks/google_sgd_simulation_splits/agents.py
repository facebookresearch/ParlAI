#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Google The Schema-Guided Dialogue(SGD) Dataset implementation for ParlAI.
"""

import os

import parlai.tasks.google_sgd_simulation_splits.build as build_
import parlai.core.tod.tod_core as tod
from parlai.core.metrics import AverageMetric
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
import parlai.core.tod.tod_agents as tod_agents
from parlai.core.opt import Opt
from parlai.tasks.google_sgd.agents import GoogleSGDParser

from typing import List, Optional


class GoogleSgdInDomainParser(GoogleSGDParser):
    """
    Overrides `__init__` and `_load_data` so that we grab our examples from our
    separately constructed custom splits.
    """

    def __init__(self, opt: Opt, shared=None):
        if shared is None:
            # full initialize the teacher as this is not a clone
            build_.build(opt)
        super().__init__(opt, shared)

    def _load_data(self, fold):
        # Best to override here because of init order
        self.dpath = os.path.join(self.opt["datapath"], "google_sgd_rl_splits")
        return super()._load_data(fold)

    def get_id_task_prefix(self):
        return "GoogleSgdInDomain"


class InDomainSystemTeacher(GoogleSgdInDomainParser, tod_agents.TodSystemTeacher):
    pass


class InDomainUserSimulatorTeacher(
    GoogleSgdInDomainParser, tod_agents.TodUserSimulatorTeacher
):
    pass


class InDomainSingleGoalAgent(GoogleSgdInDomainParser, tod_agents.TodSingleGoalAgent):
    pass


class InDomainSingleApiSchemaAgent(
    GoogleSgdInDomainParser, tod_agents.TodSingleApiSchemaAgent
):
    pass


class InDomainGoalAgent(GoogleSgdInDomainParser, tod_agents.TodGoalAgent):
    pass


class InDomainApiSchemaAgent(GoogleSgdInDomainParser, tod_agents.TodApiSchemaAgent):
    pass


class InDomainUserUttAgent(GoogleSgdInDomainParser, tod_agents.TodUserUttAgent):
    pass


class InDomainApiCallAndSysUttAgent(
    GoogleSgdInDomainParser, tod_agents.TodApiCallAndSysUttAgent
):
    pass


VALID_OUT_DOMAIN_API_NAMES = [
    "ShareLocation",
    "RequestPayment",
    "MakePayment",
    "FindApartment",
    "ScheduleVisit",
    "FindHomeByArea",
    "GetCarsAvailable",
    "ReserveCar",
]


class GoogleSgdOutDomainParser(GoogleSGDParser):
    """
    Overrides `__init__` and `_load_data` so that we grab our examples from our
    separately constructed custom splits.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("Google SGD Out Domain Parser")
        group.add_argument(
            "--filter-single-goal-episodes",
            type=bool,
            default=False,
            help="Filter for only conversations where the original has single goals",
        )

    def __init__(self, opt: Opt, shared=None):
        if shared is None:
            # full initialize the teacher as this is not a clone
            build_.build(opt)
        super().__init__(opt, shared)

    def _load_data(self, fold):
        # Best to override here because of init order
        self.dpath = os.path.join(
            self.opt["datapath"], "google_sgd_rl_splits/model_model_splits"
        )
        return super()._load_data(fold)

    def get_id_task_prefix(self):
        return "GoogleSgdOutDomain"

    def filter_goals(self, goals):
        """
        Used in single goal/api schema agents only.
        """
        result = []
        for goal in goals:
            if goal["api_name"] in VALID_OUT_DOMAIN_API_NAMES:
                result.append(goal)
        return result

    def generate_episodes(self) -> List[tod.TodStructuredEpisode]:
        data = super().generate_episodes()
        if self.opt.get("filter_single_goal_episodes"):
            result = []
            for episode in data:
                if len(episode.goal_calls_machine) == 1:
                    result.append(episode)
            data = result
        return data

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        super().custom_evaluation(teacher_action, labels, model_response)
        resp = model_response.get("text")
        if not resp:
            return

        if (
            teacher_action["type"] == tod.STANDARD_CALL
            and tod.STANDARD_API_NAME_SLOT in teacher_action["slots"]
            and teacher_action["slots"][tod.STANDARD_API_NAME_SLOT]
            in VALID_OUT_DOMAIN_API_NAMES
        ):
            if resp.startswith(tod.STANDARD_CALL):
                resp = resp[len(tod.STANDARD_CALL) :]
            predicted = tod.SerializationHelpers.str_to_api_dict(resp)
            self.metrics.add(
                f"OutDomainOnlyApis/jga",
                AverageMetric(teacher_action["slots"] == predicted),
            )


class OutDomainStandaloneApiTeacher(
    GoogleSgdOutDomainParser, tod_agents.TodStandaloneApiTeacher
):
    pass


class OutDomainSystemTeacher(GoogleSgdOutDomainParser, tod_agents.TodSystemTeacher):
    pass


class OutDomainUserSimulatorTeacher(
    GoogleSgdOutDomainParser, tod_agents.TodUserSimulatorTeacher
):
    pass


class OutDomainGoalAgent(GoogleSgdOutDomainParser, tod_agents.TodGoalAgent):
    pass


class OutDomainApiSchemaAgent(GoogleSgdOutDomainParser, tod_agents.TodApiSchemaAgent):
    pass


class OutDomainSingleGoalAgent(GoogleSgdOutDomainParser, tod_agents.TodSingleGoalAgent):
    pass


class OutDomainSingleApiSchemaAgent(
    GoogleSgdOutDomainParser, tod_agents.TodSingleApiSchemaAgent
):
    pass


class OutDomainUserUttAgent(GoogleSgdOutDomainParser, tod_agents.TodUserUttAgent):
    pass


class OutDomainApiCallAndSysUttAgent(
    GoogleSgdOutDomainParser, tod_agents.TodApiCallAndSysUttAgent
):
    pass


class OutDomainApiResponseAgent(
    GoogleSgdOutDomainParser, tod_agents.TodApiResponseAgent
):
    pass
