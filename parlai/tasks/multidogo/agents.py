#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
MultiDoGo implementation for ParlAI.

NOTE: There is still missing data in the open source version of this; implementation is not complete. See https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset/issues/1
"""

from typing import Optional
from parlai.core.params import ParlaiParser
import copy
import json
import os
from parlai.core.opt import Opt
from parlai.utils.data import DatatypeHelper
import parlai.core.tod.tod_core as tod
import parlai.core.tod.tod_agents as tod_agents
import parlai.tasks.multidogo.build as build_
from parlai.tasks.multidogo.build import get_processed_multidogo_folder
from parlai.tasks.multidogo.build import (
    DOMAINS,
    SENTENCE_INTENT,
    TURN_INTENT,
    TURN_AND_SENTENCE_INTENT,
)

INTENT_ANNOTATION_TYPES = [SENTENCE_INTENT, TURN_INTENT, TURN_AND_SENTENCE_INTENT]


class MultidogoParser(tod_agents.TodStructuredDataParser):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.add_argument(
            "--multidogo-domains",
            nargs="+",
            default=DOMAINS,
            choices=DOMAINS,
            help="Uses last passed in configuration.",
        )
        parser.add_argument(
            "--intent-type",
            default=TURN_INTENT,
            choices=INTENT_ANNOTATION_TYPES,
            help="Sets the type of intent classification labels. Sentence annotations represented as a list with adjacent entries of the same type deduped.",
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        self.fold = DatatypeHelper.fold(opt["datatype"])
        self.dpath = os.path.join(opt["datapath"], "multidogo")
        opt["datafile"] = self.fold
        build_.build(opt)
        super().__init__(opt, shared)

    def setup_episodes(self, fold):
        result = []
        domains = self.opt.get("multidogo_domains", DOMAINS)
        if type(domains) is str:
            domains = [domains]
        intent_type = self.opt.get("intent-type", TURN_INTENT)
        for _conv_id, domain, conversation in self._iterate_over_conversations(
            domains, intent_type
        ):
            if len(conversation) == 0 or not (
                all(["role" in turn for turn in conversation.values()])
            ):
                continue
            rounds = []
            prev_role = conversation["0"]["role"]
            if prev_role == "customer":
                user_utt = [conversation["0"]["text"]]
                api_call = conversation["0"].get("slots", {})
                api_resp = {}
                sys_utt = []
            else:
                user_utt = ["__SILENCE__"]
                api_call = {}
                api_resp = conversation["0"].get("slots", {})
                sys_utt = [conversation["0"]["text"]]
            all_calls = api_call
            api_call = {tod.STANDARD_API_NAME_SLOT: domain}
            for i in range(1, len(conversation)):
                turn = conversation[str(i)]
                if prev_role == "agent" and prev_role != turn["role"]:
                    rounds.append(
                        tod.TodStructuredRound(
                            user_utt="\n".join(user_utt),
                            api_call_machine=api_call,
                            api_resp_machine=api_resp,
                            sys_utt="\n".join(sys_utt),
                        )
                    )
                    user_utt = []
                    api_call = {tod.STANDARD_API_NAME_SLOT: domain}
                    api_resp = {}
                    sys_utt = []
                prev_role = turn["role"]
                slot = turn.get("slots", {})
                if prev_role == "customer":
                    user_utt.append(turn["text"])
                    api_call.update(slot)
                    all_calls.update(slot)
                else:
                    api_resp.update(slot)
                    sys_utt.append(turn["text"])

            rounds.append(
                tod.TodStructuredRound(
                    user_utt="".join(user_utt),
                    api_call_machine=api_call,
                    api_resp_machine=api_resp,
                    sys_utt="".join(sys_utt),
                )
            )
            goal_calls = copy.deepcopy(all_calls)
            goal_calls[tod.STANDARD_API_NAME_SLOT] = domain
            result.append(
                tod.TodStructuredEpisode(
                    domain=domain,
                    api_schemas_machine=[
                        {
                            tod.STANDARD_API_NAME_SLOT: domain,
                            tod.STANDARD_OPTIONAL_KEY: all_calls.keys(),
                        }
                    ],
                    goal_calls_machine=[goal_calls],
                    rounds=rounds,
                )
            )
        return result

    def _iterate_over_conversations(self, domains, intent):
        for domain in domains:
            data_folder = get_processed_multidogo_folder(
                self.dpath, domain, self.fold, intent
            )
            for filename in os.listdir(data_folder):
                if filename.endswith(".json"):
                    with open(data_folder + "/" + filename) as f:
                        data = json.load(f)
                        for conv_id, value in data.items():
                            yield conv_id, domain, value

    def get_id_task_prefix(self):
        return "Multidogo"


class SystemTeacher(MultidogoParser, tod_agents.TodSystemTeacher):
    pass


class UserSimulatorTeacher(MultidogoParser, tod_agents.TodUserSimulatorTeacher):
    pass


class DefaultTeacher(SystemTeacher):
    pass
