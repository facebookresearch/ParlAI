#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script for making light modifications to conversations from tod chats such that they are
ready for the ACUTE format.

Notably, this does things that are slightly too much of a pain in the butt to do with
regexes like "add suffixes to ids when multiple ids might have the same string" (and
change metadata appropriately).

For example, the following

```
python cleanup_conversation.py --source_file <insert filepath>_conversations.jsonl --report-path <insert filepath>.json --agent-suffixes user_utt_model _BASE_USER system_utt_model _BASE_SYSTEM --included-speakers goal_grounding_model user_utt_model system_utt_model
```

strips the API call related turns and adds "_BASE_USER" and "_BASE_SYSTEM" (which otherwise would be the model type name, ex BART) to the latter two, respecitvely.
"""

from parlai.core.params import ParlaiParser
from parlai.utils.conversations import Conversations, Metadata
from parlai.utils.io import PathManager
from parlai.core.script import ParlaiScript, register_script

from parlai.core.tod.tod_core import TodAgentType, TOD_AGENT_TYPE_TO_PREFIX

import json


@register_script("conversation_cleanup")
class ConversationCleanup(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(
            False,
            False,
            "Script for simplying conversations output from TOD. Input expected to be in conversations format, as is output",
        )
        # Following params are same as the `eval_model` script
        parser.add_argument(
            "--source-file",
            type=str,
            required=True,
            help="Source file in conversations format, generated from `tod_world_script.py`",
        )
        parser.add_argument(
            "--out-file", type=str, default=None, help="Output location."
        )
        parser.add_argument(
            "--included-speakers",
            nargs="*",
            type=str,
            choices=[e.value for e in TodAgentType],
            default=[TodAgentType.USER_UTT_AGENT, TodAgentType.SYSTEM_UTT_AGENT],
            help="Which of the speakers to not remove. Should match those in `tod_world`",
        )
        parser.add_argument(
            "--agent-suffixes",
            nargs="*",
            type=str,
            default=[
                TodAgentType.USER_UTT_AGENT,
                "_USER",
                TodAgentType.SYSTEM_UTT_AGENT,
                "_SYSTEM",
            ],
            help="List of <speaker type, suffix> pairs. Speaker type should match those in `TodAgentType`; outputs (if included) will have the suffix added to the ID. This is useful when using multiple of the same out model (ex. Bart model for both the user and the system)",
        )
        parser.add_argument(
            "--num-conversations",
            default=400,
            help="Number of conversations to include. -1 for all",
        )
        parser.add_argument(
            "--report-path",
            required=True,
            help="path of the report saved from the tod_metrics_script",
        )
        return parser

    def _get_turn_type(self, turn):
        for agent_type, prefix in TOD_AGENT_TYPE_TO_PREFIX.items():
            if prefix in turn["text"]:
                return agent_type

    def run(self):
        opt = self.opt
        if int(len(self.opt["agent_suffixes"])) % 2 != 0:
            raise RuntimeError("Agent suffix input should be even")
        suffixes = {}
        for i in range(int(len(self.opt["agent_suffixes"]) / 2)):
            agent = self.opt["agent_suffixes"][2 * i]
            suffix = self.opt["agent_suffixes"][2 * i + 1]
            suffixes[agent] = suffix

        with PathManager.open(opt["report_path"]) as r:
            report = json.load(r)["report"]
        tod_metrics = report["tod_metrics"]

        if opt["num_conversations"] > -1:
            tod_metrics = tod_metrics[: opt["num_conversations"]]

        source = self.opt["source_file"].replace(".jsonl", "")
        if self.opt["out_file"]:
            out = self.opt["out_file"]
        else:
            if (
                "conversations" in source
            ):  # just to make sure we don't overwrite anything...
                out = source.replace("conversations", "cleaned_conversations")
            else:
                out = "cleaned_" + source

        speakers = []
        with PathManager.open(out + ".jsonl", "w") as f:
            conversations = Conversations(source + ".jsonl")
            for i, conversation in enumerate(conversations):
                if opt["num_conversations"] >= 0 and i >= opt["num_conversations"]:
                    break
                cleaned_dialog = []
                for parlay_round in conversation.episode["dialog"]:
                    cleaned_parlay_round = []
                    for turn in parlay_round:
                        turn_type = self._get_turn_type(turn)
                        if turn_type in self.opt["included_speakers"]:
                            if turn_type in suffixes:
                                turn["id"] += suffixes[turn_type]
                            if turn["id"] not in speakers:
                                speakers.append(turn["id"])
                            cleaned_parlay_round.append(turn)
                    if len(cleaned_parlay_round) > 0:
                        cleaned_dialog.append(cleaned_parlay_round)
                convo = {}
                convo["dialog"] = cleaned_dialog
                convo["metadata_path"] = Metadata._get_path(out)
                convo["context"] = [
                    {
                        "synthetic_task_success": tod_metrics[i][
                            "synthetic_task_success"
                        ],
                        "goal_text": tod_metrics[i]["goal"]["text"],
                    }
                ]
                json_convo = json.dumps(convo)
                f.write(json_convo + "\n")

            old_meta = Metadata(source + ".jsonl")
            Metadata.save_metadata(
                out, old_meta.opt, old_meta.self_chat, speakers, **old_meta.extra_data
            )


if __name__ == "__main__":
    ConversationCleanup.main()
