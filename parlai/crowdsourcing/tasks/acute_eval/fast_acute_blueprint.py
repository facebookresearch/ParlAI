#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from mephisto.operations.registry import register_mephisto_abstraction
from omegaconf import MISSING

from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import (
    AcuteEvalBlueprintArgs,
    AcuteEvalBlueprint,
)

FAST_ACUTE_BLUEPRINT_TYPE = "fast_acute"


@dataclass
class FastAcuteBlueprintArgs(AcuteEvalBlueprintArgs):
    _blueprint_type: str = FAST_ACUTE_BLUEPRINT_TYPE
    _group: str = field(
        default="FastAcuteBlueprint",
        metadata={
            'help': """Run all the steps of ACUTE-Eval with one simple command"""
        },
    )
    config_path: str = field(
        default=MISSING,
        metadata={'help': 'Path to JSON of model types and their parameters'},
    )
    root_dir: str = field(default=MISSING, metadata={'help': 'Root save folder'})
    onboarding_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to JSON file of settings for running onboarding'},
    )
    models: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma separated list of models for round robin evaluation (must be at least 2)"
        },
    )
    model_pairs: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma separated list of model pairs for evaluation, model1:model2,model1:model3"
        },
    )
    acute_eval_type: str = field(
        default='engaging', metadata={"help": "Which evaluation to run for ACUTEs"}
    )
    matchups_per_pair: int = field(
        default=60,
        metadata={"help": "How many matchups to generate for each pair of models"},
    )
    task: Optional[str] = field(
        default=None, metadata={'help': 'The ParlAI task used for self-chat'}
    )
    sufficient_matchups_multiplier: int = field(
        default=2,
        metadata={
            'help': "Multiplier on how many conversation pairs to build. Probably doesn't need to be changed"
        },
    )
    num_self_chats: int = field(
        default=100, metadata={'help': "Number of self-chats to run per model"}
    )
    num_task_data_episodes: int = field(
        default=500,
        metadata={
            'help': "Number of episodes to save if running ACUTEs on a ParlAI task"
        },
    )
    selfchat_max_turns: int = field(
        default=6,
        metadata={'help': "The number of dialogue turns before self chat ends"},
    )
    use_existing_self_chat_files: bool = field(
        default=False,
        metadata={'help': "Use any existing self-chat files without prompting"},
    )


@register_mephisto_abstraction()
class FastAcuteBlueprint(AcuteEvalBlueprint):
    """
    Subclass of AcuteEvalBlueprint with params for fast ACUTE runs.
    """

    ArgsClass = FastAcuteBlueprintArgs
    BLUEPRINT_TYPE = FAST_ACUTE_BLUEPRINT_TYPE
