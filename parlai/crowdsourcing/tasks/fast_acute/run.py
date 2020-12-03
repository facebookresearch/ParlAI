#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Execute a Fast ACUTE run.

You only need to specify two arguments:
- `mephisto.blueprint.models`: the models to compare, written on the command line as
    mephisto.blueprint.models='\model1,model2\'
- `mephisto.blueprint.task`: the self-chat task
Model configurations should go in the `model_configs.py` file found in this directory.
"""

import datetime
import json
import os
import random
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import hydra
import torch
from mephisto.operations.hydra_config import RunScriptConfig, register_script_config
from mephisto.operations.operator import Operator
from mephisto.operations.registry import register_mephisto_abstraction
from mephisto.tools.scripts import load_db_and_process_config
from omegaconf import DictConfig, MISSING

from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import (
    AcuteEvalBlueprint,
    AcuteEvalBlueprintArgs,
)
from parlai.crowdsourcing.tasks.acute_eval import run
from parlai.crowdsourcing.tasks.fast_acute.analysis import (
    AcuteAnalyzer,
    setup_args as analysis_setup_args,
)
from parlai.crowdsourcing.tasks.fast_acute.util import get_hashed_combo_path
from parlai.scripts.self_chat import self_chat, setup_args as self_chat_setup_args
from parlai.utils.strings import normalize_reply
from parlai.utils.testing import capture_output


BLUEPRINT_TYPE = "fast_acute"

########################
# ACUTE EVAL CONSTANTS #
########################

ACUTE_EVAL_TYPES = {
    'human': {
        'question': 'Which speaker sounds more human?',
        's1_choice': '<Speaker 1> sounds more human',
        's2_choice': '<Speaker 2> sounds more human',
    },
    'engaging': {
        'question': 'Who would you prefer to talk to for a long conversation?',
        's1_choice': 'I would prefer to talk to <Speaker 1>',
        's2_choice': 'I would prefer to talk to <Speaker 2>',
    },
    'roleplay': {
        'question': 'How well does the speaker play their role in the conversation?',
        's1_choice': '<Speaker 1> plays their role better.',
        's2_choice': '<Speaker 2> plays their role better.',
    },
    'image': {
        "question": "Who talks about the image better?",
        "s1_choice": "<Speaker 1> talks about the image better",
        "s2_choice": "<Speaker 2> talks about the image better",
    },
}


@dataclass
class FastAcuteBlueprintArgs(AcuteEvalBlueprintArgs):
    _blueprint_type: str = BLUEPRINT_TYPE
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
    task: str = field(
        default=MISSING, metadata={'help': 'The ParlAI task used for self-chat'}
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
    selfchat_max_turns: int = field(
        default=6,
        metadata={'help': "The number of dialogue turns before self chat ends"},
    )


@register_mephisto_abstraction()
class FastAcuteBlueprint(AcuteEvalBlueprint):
    """
    Subclass of AcuteEvalBlueprint with params for fast ACUTE runs.
    """

    ArgsClass = FastAcuteBlueprintArgs
    BLUEPRINT_TYPE = BLUEPRINT_TYPE


class FastAcuteExecutor(object):
    """
    Execute fast ACUTE runs.
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
            with open(self.args.mephisto.config_path) as f:
                self.model_config = json.load(f)

        # models + task
        self._build_model_pairs()
        self.task: str = self.fast_acute_args.task

        # keep track of chat files per model
        self.chat_files: Dict[str, str] = {}

        # question config for running ACUTE
        self.question_config: Dict[str, str] = ACUTE_EVAL_TYPES[
            self.fast_acute_args.acute_eval_type
        ]

        self.run_id = self.args.mephisto.task.task_name

    #############
    # Utilities #
    #############
    def _build_model_pairs(self):
        """
        Generate self.models and self.combos from self.args.
        """
        choices = self.model_config.keys()
        combos: Set[Tuple[str, str]] = set()
        models: Set[str] = set()
        if (
            self.fast_acute_args.models is None
            and self.fast_acute_args.model_pairs is None
        ):
            raise RuntimeError(
                'Either models or model-pairs should be set for comparision.'
            )
        if self.fast_acute_args.model_pairs is not None:
            model_pairs = self.fast_acute_args.model_pairs.split(',')
            combos = [model_pair.split(':') for model_pair in model_pairs]
            for model_pair in combos:
                models.add(model_pair[0])
                models.add(model_pair[1])
        else:
            models = set(self.fast_acute_args.models.split(','))
            combos = set(combinations(models, 2))
        self.models: List[str] = list(models)
        self.models.sort()
        self.combos: List[Tuple[str, str]] = []
        for combo in combos:
            # Sort the two model names for consistency
            self.combos.append(tuple(sorted(combo)))
        # verify that models are contained in the config:
        for model in self.models:
            if model not in choices:
                raise RuntimeError(
                    f'Model {model} not specified in the config (`model_configs.py`).'
                )
        assert len(self.models) > 1, 'Must specify least 2 models'

    def _print_progress(self, msg: str):
        """
        Format a msg to print to stdout well.

        :param msg:
            message to print
        """
        print(f"\n{'-' * 60}\n {msg} \n {'-' * 60}")

    def _get_selfchat_config(self, model: str) -> Dict[str, Any]:
        """
        Return config for selfchat.

        :param model:
            model string

        :return config:
            dict config for self-chat
        """
        outfile = self._get_selfchat_log_path(model)
        config = self.model_config[model]
        config.update(
            {
                'task': self.task,
                'outfile': outfile,
                'num_self_chats': self.fast_acute_args.num_self_chats,
                'selfchat_max_turns': self.fast_acute_args.selfchat_max_turns,
                'display_examples': False,
                'log_every_n_secs': -1,
                'indent': -1,
            }
        )
        return config

    def _get_selfchat_log_path(self, model: str) -> str:
        """
        Return path to selfchat log for a given model.

        :param model:
            model string
        """
        return os.path.join(
            self.fast_acute_args.root_dir,
            f"self_chats/{model}.{self.task.replace(':', '_')}.jsonl",
        )

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
            'speakers': [model, f'other_{model}'],
        }
        dialog = dialogue_dict['dialog']
        for act_pair in dialog:
            for i, ex in enumerate(act_pair):
                if ex['id'] == 'context':
                    conversation['context'].append(ex)
                    continue
                conversation['dialogue'].append(
                    {
                        'id': model if i == 0 else f'other_{model}',
                        'text': normalize_reply(ex['text']),
                    }
                )
        return conversation

    def _load_selfchats(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load selfchats for models.

        :return conversations:
            A dictionary mapping model_id to self-chat dialogues
        """
        conversations = {}
        for m in self.models:
            model_fp = self.chat_files[m]
            conversations[m] = []
            with open(model_fp) as f_read:
                for line in f_read:
                    conversations[m].append(json.loads(line.strip()))
        return conversations

    def _assign_unique_ids(self, conversations):
        id_num = 0
        for model in self.models:
            for convo in conversations[model]:
                convo['unique_id'] = id_num
                id_num += 1

    def _build_conversation_pairs(
        self, conversations: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Build a conversation pair to show during ACUTE Eval.

        :param conversations:
            A dictionary mapping model_id to self-chat dialogues

        :return pairs:
            A list of conversation pairs
        """
        self._assign_unique_ids(conversations)
        # TODO: make it so that we don't necessarily have to evaluate
        #  every possible pairing between models
        pairs = []
        pairs_per_model = (
            self.fast_acute_args.matchups_per_pair
            * self.fast_acute_args.sufficient_matchups_multiplier
        )
        # Write random pairs of conversations
        for model_pair in self.combos:
            for _ in range(pairs_per_model):
                conversation_indices = [
                    random.choice(range(len(conversations[m]))) for m in model_pair
                ]
                pair = []
                pair_ids = []
                for i, c_id in enumerate(conversation_indices):
                    model = model_pair[i]
                    pair.append(self._acutify_convo(conversations[model][c_id], model))
                    pair_ids.append(conversations[model][c_id]['unique_id'])
                pairs.append(
                    {
                        "is_onboarding": False,
                        "speakers_to_eval": model_pair,
                        "dialogue_dicts": pair,
                        "dialogue_ids": pair_ids,
                    }
                )
        return pairs

    def _build_pairings_file(self):
        """
        Build the pairings file for the two models.
        """
        with open(os.path.join(self.fast_acute_args.root_dir, 'onboarding.json')) as f:
            onboarding_convo_pair: Dict[str, Any] = json.load(f)

        pairings_filepath = get_hashed_combo_path(
            root_dir=self.fast_acute_args.root_dir,
            subdir='pairings_files',
            task=self.task,
            combos=self.combos,
        )

        if not os.path.exists(pairings_filepath):
            self._print_progress(
                f'building pairings file, saving at {pairings_filepath}'
            )
            conversations = self._load_selfchats()
            pairs = self._build_conversation_pairs(conversations)

            with open(pairings_filepath, 'w') as f:
                # Write the onboarding convo
                f.write(json.dumps(onboarding_convo_pair) + "\n")
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
        else:
            modify_time = os.path.getmtime(pairings_filepath)
            self._print_progress(
                f'Pairings already exist {pairings_filepath}. Last modified {time.ctime(modify_time)}'
            )
            answer = ''
            while answer.lower().strip() != 'y' and answer.lower().strip() != 'o':
                answer = input('Enter y to use, o to overwrite:')
                if answer.lower().strip() == 'o':
                    self._print_progress(
                        f'building pairings file, saving at {pairings_filepath}'
                    )
                    conversations = self._load_selfchats()
                    pairs = self._build_conversation_pairs(conversations)
                    with open(pairings_filepath, 'w') as f:
                        # Write the onboarding convo
                        f.write(json.dumps(onboarding_convo_pair) + "\n")
                        for pair in pairs:
                            f.write(json.dumps(pair) + "\n")

        self._print_progress(f'loading pairings file from {pairings_filepath}')
        self.pairings_filepath = pairings_filepath

    ##################
    # Main Functions #
    ##################
    def run_selfchat(self):
        """
        Run selfchat for each model.
        """
        for model in self.models:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._print_progress(f'Running self-chat for {model}')
            outfile = self._get_selfchat_log_path(model)

            if not os.path.exists(outfile):
                config = self._get_selfchat_config(model)

                with capture_output():
                    parser = self_chat_setup_args()
                    parser.set_params(**config)
                    opt = parser.parse_args(args=[])
                self_chat(opt)

                if os.path.exists(outfile):
                    self._print_progress(f'Chats saved to {outfile} for {model}')

            self._print_progress(f'Chats already exist in {outfile}, moving on...')
            self.chat_files[model] = outfile

    def run_acute_eval(self):
        """
        Run ACUTE Eval.
        """
        self._build_pairings_file()

        total_convos = self.fast_acute_args.matchups_per_pair * len(self.combos)
        additional_params = {
            'pairings_filepath': self.pairings_filepath,
            's1_choice': self.question_config['s1_choice'],
            's2_choice': self.question_config['s2_choice'],
            'eval_question': self.question_config['question'],
            'num_matchup_pairs': total_convos,
            'block_qualification': f"{self.args.mephisto.task.task_name}_block_qual",
        }
        overwritten_param_strings = []
        for key, val in additional_params.items():
            if val != self.fast_acute_args.get(key, None):
                self.fast_acute_args[key] = val
                overwritten_param_strings.append(f'\t{key}: {val}')
        if len(overwritten_param_strings) > 0:
            overwritten_param_output = '\n'.join(overwritten_param_strings)
            self._print_progress(
                f"The following ACUTE-Eval parameters will be overwritten to the following:\n"
                f"{overwritten_param_output}"
            )
        db, cfg = load_db_and_process_config(self.args)
        operator = Operator(db)
        operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
        operator.wait_for_runs_then_shutdown(
            skip_input=True, log_rate=cfg.monitoring_log_rate
        )

    def analyze_results(self):
        """
        Analyze results of ACUTE Eval run.

        Save results to appropriate filepath.
        """
        self._print_progress(f'Analyzing Results for run id {self.run_id}')
        parser = analysis_setup_args()
        opt = parser.parse_args([])
        today = datetime.date.today().isoformat()
        self.results_path = get_hashed_combo_path(
            root_dir=self.fast_acute_args.root_dir,
            subdir=f'acute_results/{today}/',
            task=self.task,
            combos=self.combos,
        )
        opt.update(
            {
                'model_strings': ','.join(self.models),
                'run_id': self.run_id,
                'root_dir': self.fast_acute_args.root_dir,
                'outdir': self.results_path,
                'task': self.task,
            }
        )

        analyzer = AcuteAnalyzer(opt, self.run_id)
        self.results = analyzer.get_matchup_totals_with_significance()
        analyzer.save_results()

        self._print_progress(f'ACUTE Results: {self.results}')
        self._print_progress(f'ACUTE results saved to {self.results_path}')


TASK_DIRECTORY = os.path.dirname(os.path.abspath(run.__file__))
# Read in any task config JSON/HTML files from the ACUTE-Eval directory

defaults = [
    {"mephisto/blueprint": BLUEPRINT_TYPE},
    {"mephisto/architect": "local"},
    {"mephisto/provider": "mock"},
    'conf/base',
    {"conf": "example"},
]


@dataclass
class TestScriptConfig(RunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY
    current_time: int = int(time.time())
    monitoring_log_rate: int = field(
        default=30,
        metadata={
            'help': 'Frequency in seconds of logging the monitoring of the crowdsourcing task'
        },
    )


register_script_config(name='fast_acute_scriptconfig', module=TestScriptConfig)


@hydra.main(config_name="fast_acute_scriptconfig")
def main(cfg: DictConfig) -> None:

    runner = FastAcuteExecutor(cfg)

    # Create self-chats
    runner.run_selfchat()

    # Run ACUTE-Eval
    runner.run_acute_eval()

    # Analyze the results
    runner.analyze_results()


if __name__ == '__main__':
    main()
