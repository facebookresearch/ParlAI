#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Taskmaster-3 implementation for ParlAI.

No official train/valid/test splits are available as of 2020-05-18, so we make our own
splits.
"""

from parlai.core.params import ParlaiParser
import os
import pandas as pd
from parlai.core.opt import Opt
import parlai.core.tod.tod_core as tod
import json
from typing import Optional
from parlai.utils.data import DatatypeHelper
from parlai.utils.io import PathManager

import parlai.tasks.taskmaster3.build as build_
import parlai.core.tod.tod_agents as tod_agents

SILENCE_TOKEN = "__SILENCE__"


class Taskmaster3Parser(tod_agents.TodStructuredDataParser):
    """
    Abstract data loader.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        return super().add_cmdline_args(parser, partial_opt)

    def __init__(self, opt: Opt, shared=None):
        self.fold = DatatypeHelper.fold(opt["datatype"])
        opt["datafile"] = self.fold
        self.dpath = os.path.join(opt["datapath"], "taskmaster-3")
        build_.build(opt)
        super().__init__(opt, shared)

    def _load_data(self, fold):
        # load up the ontology
        fn = os.path.join(self.dpath, "apis.json")
        with PathManager.open(fn, "r") as f:
            api_schema_raw = json.load(f)

        chunks = []
        for section in range(20):
            section = f"{section:02d}"
            with PathManager.open(
                os.path.join(self.dpath, f"data_{section}.json")
            ) as f:
                subset = pd.read_json(f)
            chunks.append(subset)
        chunks = pd.concat(chunks, axis=0)
        # deterministic shuffle data for splits
        chunks = chunks.sample(frac=1.0, random_state=42)
        split_size = len(chunks) // 10
        if fold == "train":
            chunks = chunks[: split_size * 8]
        elif fold == "valid":
            chunks = chunks[split_size * 8 : split_size * 9]
        elif fold == "test":
            chunks = chunks[split_size * 9 :]
        return chunks, api_schema_raw

    def _parse_apis_to_slots(self, apis_blob):
        calls = []
        responses = []
        for api in apis_blob:
            call = api["args"]
            call[tod.STANDARD_API_NAME_SLOT] = api["name"]
            calls.append(call)
            response = api["response"]
            response[tod.STANDARD_API_NAME_SLOT] = api["name"]
            responses.append(response)
        return calls, responses

    def _get_utterance_and_apis_for_speaker(self, speaker, utterances, idx):
        utts = []
        calls = []
        responses = []
        while idx < len(utterances):
            here = utterances[idx]
            if here["speaker"] != speaker:
                break
            utts.append(here["text"])
            here_calls, here_responses = self._parse_apis_to_slots(here.get("apis", []))
            calls += here_calls
            responses += here_responses
            idx += 1
        return idx, "\n".join(utts), calls, responses

    def _parse_to_api_schema(self, raw):
        result = []
        for key, val in raw.items():
            here = {}
            here[tod.STANDARD_API_NAME_SLOT] = key
            req = val.get("args", {}).get("all_of", [])
            opt = val.get("args", {}).get("any_of", [])
            if len(req) > 0:
                here[tod.STANDARD_REQUIRED_KEY] = req
            if len(opt) > 0:
                here[tod.STANDARD_OPTIONAL_KEY] = opt
            result.append(here)
        return result

    def _get_turns_from_parsed(self, user_utt, api_calls, api_resps, sys_utt):
        assert len(api_calls) == len(api_resps)
        if len(api_calls) == 0:
            api_calls = [{}]
            api_resps = [{}]
        turns = len(api_calls)
        user_utts = [SILENCE_TOKEN] * turns
        user_utts[0] = user_utt
        sys_utts = [SILENCE_TOKEN] * turns
        sys_utts[turns - 1] = sys_utt

        result = []
        for i in range(turns):
            result.append(
                tod.TodStructuredRound(
                    sys_utt=sys_utts[i],
                    api_call_machine=api_calls[i],
                    api_resp_machine=api_resps[i],
                    user_utt=user_utts[i],
                )
            )
        return result

    def setup_episodes(self, fold):
        """
        Parses into TodStructuredEpisode.
        """
        chunks, api_schema_raw = self._load_data(fold)
        api_schemas_machine = self._parse_to_api_schema(api_schema_raw)
        episodes = []
        for _, row in chunks.iterrows():
            utterances = row["utterances"][:]
            idx = 0
            rounds = []
            goal_calls = []
            if len(utterances) > 0 and utterances[0]["speaker"] == "assistant":
                (
                    idx,
                    sys_utt,
                    api_call,
                    api_resp,
                ) = self._get_utterance_and_apis_for_speaker(
                    "assistant", utterances, idx
                )

                turns = self._get_turns_from_parsed(
                    SILENCE_TOKEN, api_call, api_resp, sys_utt
                )
                for t in turns:
                    rounds.append(t)

            while idx < len(utterances):
                (
                    idx,
                    user_utt,
                    calls_user,
                    responses_user,
                ) = self._get_utterance_and_apis_for_speaker("user", utterances, idx)
                (
                    idx,
                    sys_utt,
                    calls_system,
                    responses_system,
                ) = self._get_utterance_and_apis_for_speaker(
                    "assistant", utterances, idx
                )
                api_calls = calls_user + calls_system
                api_resps = responses_user + responses_system
                goal_calls += api_calls
                turns = self._get_turns_from_parsed(
                    user_utt, api_calls, api_resps, sys_utt
                )
                for t in turns:
                    rounds.append(t)

            episode = tod.TodStructuredEpisode(
                api_schemas_machine=api_schemas_machine,
                goal_calls_machine=goal_calls,
                rounds=rounds,
                delex=self.opt.get("delex", False),
            )
            episodes.append(episode)
        return episodes

    def get_id_task_prefix(self):
        return "Taskmaster3"

    def _label_fold(self, chunks):
        return chunks.conversation_id.apply(self._h)


class SystemTeacher(Taskmaster3Parser, tod_agents.TodSystemTeacher):
    pass


class UserSimulatorTeacher(Taskmaster3Parser, tod_agents.TodUserSimulatorTeacher):
    pass


class DefaultTeacher(SystemTeacher):
    pass
