#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Execute a Fast ACUTE run with Q-learning.

Run mturk/tasks/q_function/scripts/compile_results.py to parse the latest logs. Model
configurations should go in the `model_configs.py` file found in this directory.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import hydra
from mephisto.operations.hydra_config import register_script_config
from mephisto.operations.registry import register_mephisto_abstraction
from omegaconf import DictConfig, MISSING

import parlai.crowdsourcing.tasks.fast_acute.run as acute_eval
from parlai.crowdsourcing.tasks.acute_eval import run
from parlai.crowdsourcing.tasks.fast_acute.run import (
    FastAcuteBlueprint,
    FastAcuteBlueprintArgs,
    FastAcuteExecutor,
)
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig
from parlai.utils.strings import normalize_reply

BLUEPRINT_TYPE = "fast_acute_q_function"


@dataclass
class FastAcuteQFunctionBlueprintArgs(FastAcuteBlueprintArgs):
    _blueprint_type: str = BLUEPRINT_TYPE
    _group: str = field(
        default="FastAcuteQFunctionBlueprint",
        metadata={'help': "Execute a fast ACUTE run with Q-learning"},
    )
    config_path: str = field(
        default=MISSING,
        metadata={'help': 'Path to JSON of model types and their parameters'},
    )


@register_mephisto_abstraction()
class FastAcuteQFunctionBlueprint(FastAcuteBlueprint):
    """
    Subclass of FastAcuteBlueprint with Q-function conversations.
    """

    ArgsClass = FastAcuteQFunctionBlueprintArgs
    BLUEPRINT_TYPE = BLUEPRINT_TYPE


class QLearningFastAcuteExecutor(FastAcuteExecutor):
    """
    Execute fast ACUTE runs with Q-function learning.
    """

    def __init__(self, args: DictConfig, model_config: Optional[Dict[str, Any]] = None):
        """
        Pass in model_config directly to override the model config file,
        args.mephisto.blueprint, that would be read in otherwise.
        """

        self.args = args
        self.fast_acute_args = self.args.mephisto.blueprint

        # Load configs for models
        if model_config is not None:
            self.model_config = model_config
        else:
            with open(self.fast_acute_args.config_path) as f:
                self.model_config = json.load(f)

        # models + task
        self._build_model_pairs()

        # keep track of chat files per model
        self.chat_files: Dict[str, str] = {}
        for model in self.models:
            self.chat_files[model] = self.model_config[model]['log_path']

        self.task: str = 'q'
        # question config for running ACUTE
        self.question_config: Dict[str, str] = acute_eval.ACUTE_EVAL_TYPES[
            self.fast_acute_args.acute_eval_type
        ]
        # prepare 2x convo pairs so we don't run out of them (the same logic as in _build_conversation_pairs in fast_acute/fast_eval.py)
        # The logic of calculating num_matchup_pairs and num_conversations in acute_args is the same as that in fast_eval/fast_acute.py therefore hidden here.

        self.run_id = self.args.mephisto.task.task_name

    def _acutify_convo(
        self, dialogue_dict: Dict[str, Any], model: str
    ) -> Dict[str, List]:
        """
        Format world-logged conversation to be ACUTE format.

        :param dialogue_dict:
            dictionary containing the dialogue for a model
        :param model:
            model string

        :return conversation:
            An ACUTE-Readable conversation
        """
        conversation = {
            'context': [],
            'dialogue': [],
            'speakers': ['human_evaluator', model],
        }
        if (
            'is_selfchat' in self.model_config[model]
            and self.model_config[model]['is_selfchat']
        ):
            if 'flip' in self.model_config[model]:
                conversation['speakers'] = ['other_speaker', model]
            else:
                conversation['speakers'] = [model, 'other_speaker']
        dialog = dialogue_dict['dialog']
        for act_pair in dialog:
            for i, ex in enumerate(act_pair):
                if ex['id'] == 'context':
                    conversation['context'].append(ex)
                    continue
                else:
                    # from Mary's log agent 1 is the model, agent 0 is human
                    convo = {'id': ex['id'], 'text': normalize_reply(ex['text'])}
                    if (
                        'is_selfchat' in self.model_config[model]
                        and self.model_config[model]['is_selfchat']
                    ):
                        # if_selfchat override agent_id
                        if 'flip' in self.model_config[model]:
                            if i % 2 == 1:
                                convo['id'] = model
                            else:
                                convo['id'] = 'other_speaker'
                        else:
                            if i % 2 == 0:
                                convo['id'] = model
                            else:
                                convo['id'] = 'other_speaker'
                    conversation['dialogue'].append(convo)

        return conversation


ACUTE_EVAL_TASK_DIRECTORY = os.path.dirname(os.path.abspath(run.__file__))
# Read in any task config JSON/HTML files from the ACUTE-Eval directory

defaults = [
    {"mephisto/blueprint": BLUEPRINT_TYPE},
    {"mephisto/architect": "local"},
    {"mephisto/provider": "mock"},
    'conf/base_q_function',
    {"conf": "example_q_function"},
]


@dataclass
class TestScriptConfig(MTurkRunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = ACUTE_EVAL_TASK_DIRECTORY
    monitoring_log_rate: int = field(
        default=30,
        metadata={
            'help': 'Frequency in seconds of logging the monitoring of the crowdsourcing task'
        },
    )


register_script_config(name='fast_acute_q_function', module=TestScriptConfig)


@hydra.main(config_name="fast_acute_q_function")
def main(cfg: DictConfig) -> None:

    runner = QLearningFastAcuteExecutor(cfg)

    # Run ACUTE-Evals
    runner.run_acute_eval()

    # Analyze the results
    runner.analyze_results()


if __name__ == '__main__':
    main()
