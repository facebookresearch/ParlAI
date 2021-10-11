#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Teachers for TOOD.
"""
from parlai.core.metrics import AverageMetric
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.core.params import ParlaiParser

import parlai.tod.tod_core as tod
from parlai.tod.tod_core import SerializationHelpers
from parlai.tod.tod_agents import TodStructuredDataAgent

from typing import Optional


class SystemTeacher(TodStructuredDataAgent, DialogTeacher):
    """
    TOD agent teacher which produces both API calls and NLG responses.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            "--api-descriptions",
            type="bool",
            default=False,
            help="Preempt first turn with intents + required/optional parameters as key/value for given domain",
        )
        parser.add_argument(
            "--standalone-api",
            type="bool",
            default=True,
            help="Noop for this agent. Included to make sweeps easier.",
        )
        parser.add_argument(
            "--api-jga-record",
            type=bool,
            default=True,
            help="Should we save jga information per api schema?",
        )
        parser.add_argument(
            "--domain-jga-record",
            type=bool,
            default=False,
            help="Should we save jga information per domain?",
        )

        return parser

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        resp = model_response.get("text")
        if not resp:
            return
        teacher_domains = []
        domain = None
        if self.opt["domain_jga_record"]:
            domain = teacher_action["domain"]
            teacher_domains = [domain]

        if teacher_action["type"] == tod.STANDARD_CALL:
            if resp.startswith(tod.STANDARD_CALL):
                resp = resp[len(tod.STANDARD_CALL) :]
            predicted = SerializationHelpers.str_to_api_dict(resp)

            metrics = tod.SlotMetrics(
                teacher_slots=teacher_action["slots"],
                predicted_slots=predicted,
                avg_jga_nlg_bleu=True,
                domain=domain,
            ).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

            if self.opt["api_jga_record"] and len(teacher_action["slots"]) > 0:
                teacher = teacher_action["slots"]
                slots = list(teacher.keys())
                slots.remove(tod.STANDARD_API_NAME_SLOT)
                api_here = (
                    "api-"
                    + teacher[tod.STANDARD_API_NAME_SLOT]
                    + "--"
                    + "-".join(slots)
                )
                self.metrics.add(f"{api_here}/jga", AverageMetric(teacher == predicted))

        elif teacher_action["type"] == tod.STANDARD_SYSTEM_UTTERANCE:
            metrics = tod.NlgMetrics(
                guess=resp,
                labels=labels,
                teacher_domains=teacher_domains,
                avg_jga_nlg_bleu=True,
            ).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

    def setup_data(self, fold):
        for episode in self.generate_episodes():
            if self.opt.get("api_descriptions"):
                descriptions = episode.api_descriptions_utt
            else:
                descriptions = ""
            yield {
                "text": f"{tod.STANDARD_API_DESCRIPTIONS}{descriptions}",
                "label": f"{tod.STANDARD_API_DESCRIPTIONS}",
                "domain": episode.domain,
                "type": tod.STANDARD_API_DESCRIPTIONS,
                "slots": {},
            }, True
            for r in episode.rounds:
                yield {
                    "text": f"{tod.STANDARD_USER_UTTERANCE}{r.user_utt}",
                    "label": f"{tod.STANDARD_CALL}{r.api_call_utt}",
                    "domain": episode.domain,
                    "type": tod.STANDARD_CALL,
                    "slots": r.api_call_machine,
                }, False
                yield {
                    "text": f"{tod.STANDARD_RESP}{r.api_resp_utt}",
                    "label": f"{tod.STANDARD_SYSTEM_UTTERANCE}{r.sys_utt}",
                    "domain": episode.domain,
                    "slots": r.api_resp_machine,
                    "type": tod.STANDARD_SYSTEM_UTTERANCE,
                }, False

    def get_agent_type_suffix(self):
        return "SystemTeacher"


class UserSimulatorTeacher(TodStructuredDataAgent, DialogTeacher):
    """
    Teacher that tries to simulate user actions (ie, switches text/labels between USER
    and SYSTEM)
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            "--api-descriptions",
            type="bool",
            default=False,
            help="Noop for this teacher. Included here mostly to make sweeps easy",
        )
        parser.add_argument(
            "--standalone-api",
            type="bool",
            default=True,
            help="Use a separately constructed class to handle API calls, in contrast to having the 'user' model also generate API responses.",
        )
        return parser

    def setup_data(self, fold):
        for episode in self.generate_episodes():
            if len(episode.rounds) < 1:
                continue
            yield {
                "text": f"{tod.STANDARD_GOAL}{episode.goal_calls_utt}",
                "label": f"{tod.STANDARD_USER_UTTERANCE}{episode.rounds[0].user_utt}",
                "domain": episode.domain,
                "type": tod.STANDARD_USER_UTTERANCE,
            }, True
            for i, r in enumerate(episode.rounds):
                if i == len(episode.rounds) - 1:
                    continue
                if not self.opt.get("standalone_api", True):
                    yield {
                        "text": f"{tod.STANDARD_CALL}{r.api_call_utt}",
                        "label": f"{tod.STANDARD_RESP}{r.api_resp_utt}",
                        "domain": episode.domain,
                        "slots": r.api_resp_machine,
                        "type": tod.STANDARD_RESP,
                    }, False
                yield {
                    "text": f"{tod.STANDARD_SYSTEM_UTTERANCE}{r.sys_utt}",
                    "label": f"{tod.STANDARD_USER_UTTERANCE}{episode.rounds[i+1].user_utt}",
                    "domain": episode.domain,
                    "type": tod.STANDARD_USER_UTTERANCE,
                    "slots": {},  # slots in agent/user turns are meaningless
                }, False

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        resp = model_response.get("text")
        if not resp:
            return
        if (
            not self.opt.get("standalone_api")
            and teacher_action["type"] == tod.STANDARD_RESP
        ):
            if resp.startswith(tod.STANDARD_RESP):
                resp = resp[len(tod.STANDARD_RESP) :]
            predicted = SerializationHelpers.str_to_api_dict(resp)

            metrics = tod.SlotMetrics(teacher_action["slots"], predicted).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

        elif teacher_action["type"] == tod.STANDARD_USER_UTTERANCE:
            metrics = tod.NlgMetrics(resp, labels).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

    def get_agent_type_suffix(self):
        return "UserSimulatorTeacher"


class TodStandaloneApiTeacher(TodStructuredDataAgent, DialogTeacher):
    def setup_data(self, fold):
        # As a default, just put everything in
        for fold_overwrite in ["train", "valid", "test"]:
            for episode in self.setup_episodes(fold_overwrite):
                first = True
                for r in episode.rounds:
                    if len(r.api_call_machine) > 0:
                        yield {
                            "text": f"{tod.STANDARD_CALL}{r.api_call_utt}",
                            "label": f"{tod.STANDARD_RESP}{r.api_resp_utt}",
                            "id": self.id,
                            "domain": episode.domain,
                        }, first
                        first = False

    def get_agent_type_suffix(self):
        return "StandaloneApiTeacher"
