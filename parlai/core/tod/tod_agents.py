#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agents (used for dumping data) and Teachers (for training models) related to the TOD
conversation setup.

As a convention, agents and teachers that are inheritable are prefixed with "Tod"
whereas those that can be used as-is are not. Similarly, classes and functions that do
not need to be exposed outside of this file are prefixed with a single underscore ('_')
"""

from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.teachers import DialogTeacher
from parlai.utils.distributed import is_distributed, get_rank, num_workers

import parlai.core.tod.tod_core as tod
from parlai.core.tod.tod_core import SerializationHelpers
from parlai.core.tod.teacher_metrics import SlotMetrics, NlgMetrics

from typing import Optional, List
import json
import pickle
import difflib
import random
from math import ceil


######### Agents that dump information from a dataset; base classes
class TodStructuredDataParser(Agent):
    """
    Base class that specifies intermediate representations for Tod conversations.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        if hasattr(super(), "add_cmdline_args"):
            parser = super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("TOD StructuredData agent")
        group.add_argument(
            "--episodes-randomization-seed",
            type=int,
            default=-1,
            help="Randomize episodes in a predictable way (eg, for few shot). Set to -1 for no randomization. ",
        )
        parser.add_argument(
            "--n-shot",
            default=-1,
            type=int,
            help="Number of dialogues to keep for each of train/valid/test. -1 means all. Dialogues of lower numbers are strict subsets of larger numbers. Do not use in conjunction with `--percent-shot`. Use `--episodes-randomization-seed` to change seed. NOTE: Beware of using this flag when multitasking as this will apply to *all* datasets unless the ':' syntax for specifying per-dataset flags is used.",
        )
        parser.add_argument(
            "--percent-shot",
            default=-1,
            type=float,
            help="Percentage of dialogues to keep for each of train/valid/test. -1 means all. Dialogues of lower numbers are strict subsets of larger numbers. Do not use in conjunction with `--n-shot`. Use `--episodes-randomization-seed` to change seed. NOTE: Beware of using this flag when multitasking as this will apply to *all* datasets unless the ':' syntax for specifying per-dataset flags is used.",
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.id = self.get_id_task_prefix() + "_" + self._get_agent_type_suffix()
        if shared is None:
            self.episodes = self.generate_episodes()
        else:
            self.episodes = shared["episodes"]

    def share(self):
        share = super().share()
        share["episodes"] = self.episodes
        return share

    def setup_episodes(self, fold: str) -> List[tod.TodStructuredEpisode]:
        """
        Fold here is a data fold.
        """
        raise NotImplementedError(
            "Must have method for generating an episode. Must be set in downstream Parser for a given task"
        )

    def generate_episodes(self) -> List[tod.TodStructuredEpisode]:
        if self.opt.get("n_shot", -1) >= 0 and self.opt.get("percent_shot", -1) >= 0:
            # Validate before spending a while to load eeverything
            raise RuntimeError("Both `--n-shot` and `--percent-shot` in use!")
        episodes = list(self.setup_episodes(self.fold))
        if self.opt.get("episodes_randomization_seed", -1) != -1:
            random.Random(self.opt["episodes_randomization_seed"]).shuffle(episodes)
        if self.opt.get("n_shot", -1) != -1:
            episodes = episodes[: self.opt["n_shot"]]
        elif self.opt.get("percent_shot", -1) >= 0:
            episodes = episodes[: int(len(episodes) * self.opt["percent_shot"])]
        return episodes

    def get_id_task_prefix(self) -> str:
        """
        Convenience for setting IDs.
        """
        raise NotImplementedError(
            "Must set ID prefix in downstream task agent. Must be set in downsream Parser for a given task"
        )

    def _get_agent_type_suffix(self) -> str:
        """
        Convenience for setting IDs.
        """
        raise NotImplementedError(
            "Must set in downstream agent within `tod_agents`. If you see this error, something is wrong with TOD Infrastructure"
        )


############# Teachers
class TodSystemTeacher(TodStructuredDataParser, DialogTeacher):
    """
    TOD agent teacher which produces both API calls and NLG responses.

    First turn is API Schema grounding, which may be a an empty schema.
    Subsequent turns alternate between
        1. User utterance -> API Call
        2. API Response -> System Utterance
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            "--api-schemas",
            type="bool",
            default=False,
            help="Preempt first turn with intents + required/optional parameters as key/value for given domain",
        )
        parser.add_argument(
            "--api-jga-record",
            type=bool,
            default=True,
            help="Breaks out jga into individual api schemas",
        )
        parser.add_argument(
            "--domain-jga-record",
            type=bool,
            default=False,
            help="Breaks out jga into individual domains",
        )
        parser.add_argument(
            "--domain-nlg-record",
            type=bool,
            default=False,
            help="Breaks out nlg into individual domains",
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self._num_examples_cache = sum([len(x.rounds) * 2 for x in self.episodes])
        self._num_episodes_cache = len(self.episodes)

    def custom_evaluation(
        self, teacher_action: Message, labels, model_response: Message
    ):
        resp = model_response.get("text")
        if not resp:
            return
        if teacher_action["type"] == tod.STANDARD_CALL:
            if resp.startswith(tod.STANDARD_CALL):
                resp = resp[len(tod.STANDARD_CALL) :]
            predicted = SerializationHelpers.str_to_api_dict(resp)
            domains = (
                [teacher_action["domain"]] if self.opt["domain_jga_record"] else []
            )

            metrics = SlotMetrics(
                teacher_slots=teacher_action["slots"],
                predicted_slots=predicted,
                prefixes=domains,
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
            domains = (
                [teacher_action["domain"]] if self.opt["domain_nlg_record"] else []
            )
            metrics = NlgMetrics(guess=resp, labels=labels, prefixes=domains).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

    def setup_data(self, fold):
        for episode in self.generate_episodes():
            if self.opt.get("api_schemas"):
                schemas = episode.api_schemas_utt
            else:
                schemas = ""
            yield {
                "text": f"{tod.STANDARD_API_SCHEMAS}{schemas}",
                "label": f"{tod.STANDARD_API_SCHEMAS}",
                "domain": episode.domain,
                "type": tod.STANDARD_API_SCHEMAS,
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

    def _get_agent_type_suffix(self):
        return "SystemTeacher"


class TodUserSimulatorTeacher(TodStructuredDataParser, DialogTeacher):
    """
    Teacher that has `Goal->User Utterance` for its first turn, then `System
    Utterance->User Utterance` for all subsequent turns.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        # Manually set number of examples + number of episodes
        self._num_examples_cache = sum([len(x.rounds) for x in self.episodes])
        self._num_episodes_cache = len(self.episodes)

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
        if teacher_action["type"] == tod.STANDARD_RESP:
            if resp.startswith(tod.STANDARD_RESP):
                resp = resp[len(tod.STANDARD_RESP) :]
            predicted = SerializationHelpers.str_to_api_dict(resp)

            metrics = SlotMetrics(teacher_action["slots"], predicted).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

        elif teacher_action["type"] == tod.STANDARD_USER_UTTERANCE:
            metrics = NlgMetrics(resp, labels).report()
            for key, value in metrics.items():
                self.metrics.add(key, value)

    def _get_agent_type_suffix(self):
        return "UserSimulatorTeacher"


class TodStandaloneApiTeacher(TodStructuredDataParser, DialogTeacher):
    """
    Use this to generate a database for `StandaloneApiAgent`.

    Set this as the teacher with `StandaloneApiAgent` as the agent. Ex for a MultiWoz
    V2.2 standalone API, use ``` parlai train -t multiwoz_v22:StandaloneApiTeacher -m
    parlai.core.tod.tod_agents:StandaloneApiAgent -eps 4 -mf output ```
    """

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

    def _get_agent_type_suffix(self):
        return "StandaloneApiTeacher"
