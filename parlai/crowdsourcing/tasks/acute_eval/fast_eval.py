#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Execute a Fast ACUTE run.
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
from mephisto.operations.hydra_config import register_script_config
from mephisto.operations.operator import Operator
from mephisto.tools.scripts import load_db_and_process_config
from omegaconf import DictConfig

from parlai.crowdsourcing.tasks.acute_eval.analysis import (
    AcuteAnalyzer,
    setup_args as analysis_setup_args,
)
from parlai.crowdsourcing.tasks.acute_eval.dump_task_to_acute_format import (
    dump_data as convert_task_data,
    setup_args as convert_task_setup_args,
)
from parlai.crowdsourcing.tasks.acute_eval.fast_acute_blueprint import (
    FAST_ACUTE_BLUEPRINT_TYPE,
)
from parlai.crowdsourcing.tasks.acute_eval.util import (
    get_hashed_combo_path,
    TASK_DIRECTORY,
)
from parlai.crowdsourcing.utils.mturk import MTurkRunScriptConfig
from parlai.scripts.self_chat import self_chat, setup_args as self_chat_setup_args
from parlai.utils.strings import normalize_reply
from parlai.utils.testing import capture_output

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


class FastAcuteExecutor(object):
    """
    Execute fast ACUTE runs.
    """

    ANALYZER = AcuteAnalyzer

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
                raise RuntimeError(f'Model {model} not specified in the config.')
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

    def _get_task_conversion_config(self, model: str) -> Dict[str, Any]:
        """
        Return config for task conversion to conversation format.
        """
        outfile = self._get_task_data_path(model)
        config = self.model_config[model]
        config.update(
            {
                'outfile': outfile,
                'num_episodes': self.fast_acute_args.num_task_data_episodes,
                'speaker_0_id': f'{model}_as_human',
                'speaker_1_id': model,
            }
        )
        return config

    @staticmethod
    def get_relative_selfchat_log_path(root_dir: str, model: str, task: str) -> str:
        """
        Return path to selfchat log for a given model, given inputs.

        Useful for getting selfchat log path without instantiating the exector.
        """
        self_chats_folder = os.path.join(root_dir, 'self_chats')
        os.makedirs(self_chats_folder, exist_ok=True)
        return os.path.join(
            self_chats_folder, f"{model}.{task.replace(':', '_')}.jsonl"
        )

    def _get_log_path(self, model: str) -> str:
        """
        Return path to chat logs for the given model.
        """
        config = self.model_config[model]
        if 'log_path' in config:
            path = config['log_path']
            assert os.path.exists(
                path
            ), f'Path provided in log_path for {model} does not exist'
        elif 'task' in config:
            path = self._get_task_data_path(model)
        elif 'model' in config:
            path = self._get_selfchat_log_path(model)
        else:
            raise ValueError(f'Invalid config for {model}')

        return path

    def _get_task_data_path(self, model: str) -> str:
        """
        Return path to task data as conversations for given task.
        """
        task_data_dir = os.path.join(
            self.fast_acute_args.root_dir, 'tasks_as_conversations'
        )
        os.makedirs(task_data_dir, exist_ok=True)
        return os.path.join(task_data_dir, f"{model}.jsonl")

    def _get_selfchat_log_path(self, model: str) -> str:
        """
        Return path to selfchat log for a given model.

        :param model:
            model string
        """
        return self.get_relative_selfchat_log_path(
            root_dir=self.fast_acute_args.root_dir, model=model, task=self.task
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
        is_selfchat = 'model' in self.model_config[model] or self.model_config[
            model
        ].get('is_selfchat', False)
        # It's a self-chat if one of the following are true:
        # (1) a model is specified in the config, meaning that we're collecting
        #   self-chats with that model
        # (2) we manually set 'is_selfchat' to True in the config
        if is_selfchat:
            # Set which speaker we will evaluate the conversation turns of
            speaker_idx = self.model_config[model].get('speaker_idx', 0)
            assert speaker_idx in [0, 1]
        conversation = {'context': [], 'dialogue': [], 'speakers': []}
        dialog = dialogue_dict['dialog']
        for act_pair in dialog:
            for i, ex in enumerate(act_pair):
                if ex['id'] == 'context':
                    conversation['context'].append(ex)
                    continue
                if is_selfchat:
                    speaker_id = model if i == speaker_idx else f'other_speaker'
                else:
                    speaker_id = ex['id']
                if speaker_id not in conversation['speakers']:
                    conversation['speakers'].append(speaker_id)
                conversation['dialogue'].append(
                    {'id': speaker_id, 'text': normalize_reply(ex['text'])}
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
        # Write pairs of conversations
        for model_pair in self.combos:
            for i in range(pairs_per_model):
                if self.fast_acute_args.randomize_conversations:
                    conversation_indices = [
                        random.choice(range(len(conversations[m]))) for m in model_pair
                    ]
                else:
                    conversation_indices = [i for _ in model_pair]
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

    def _build_pairings_file(self, pairings_filepath: str):
        """
        Build and save pairings to pairings file.
        """

        if self.fast_acute_args.onboarding_path is not None:
            onboarding_path = self.fast_acute_args.onboarding_path
        else:
            # Default onboarding location
            onboarding_path = os.path.join(
                self.fast_acute_args.root_dir, 'onboarding.json'
            )
        with open(onboarding_path) as f:
            onboarding_convo_pair: Dict[str, Any] = json.load(f)

        self._print_progress(f'building pairings file, saving at {pairings_filepath}')
        conversations = self._load_selfchats()
        pairs = self._build_conversation_pairs(conversations)

        with open(pairings_filepath, 'w') as f:
            # Write the onboarding convo
            f.write(json.dumps(onboarding_convo_pair) + "\n")
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

    def _load_pairings_file(self):
        """
        Build the pairings file for the two models.

        If a pairings file already exists, we ask the user whether they would like to
        overwrite the pairings file.
        """
        pairings_filepath = get_hashed_combo_path(
            root_dir=self.fast_acute_args.root_dir,
            subdir='pairings_files',
            task=self.task,
            combos=self.combos,
        )
        if not os.path.exists(pairings_filepath):
            self._build_pairings_file(pairings_filepath)
        else:
            modify_time = os.path.getmtime(pairings_filepath)
            self._print_progress(
                f'Pairings already exist {pairings_filepath}. Last modified {time.ctime(modify_time)}'
            )
            if not self.fast_acute_args.use_existing_self_chat_files:
                answer = ''
                while answer.lower().strip() != 'y' and answer.lower().strip() != 'o':
                    answer = input('Enter y to use, o to overwrite:')
                    if answer.lower().strip() == 'o':
                        self._build_pairings_file(pairings_filepath)

        self._print_progress(f'loading pairings file from {pairings_filepath}')
        self.pairings_filepath = pairings_filepath

    def _convert_task_to_conversations(self, model: str):
        """
        Convert task data to conversations format.
        """
        self._print_progress(
            f'Converting task data to conversations format for {model}'
        )
        config = self._get_task_conversion_config(model)

        with capture_output():
            parser = convert_task_setup_args()
            parser.set_params(**config)
            opt = parser.parse_args(args=[])
        convert_task_data(opt)

    ##################
    # Main Functions #
    ##################
    def compile_chat_logs(self):
        """
        Compile chat logs.

        Logs are generated depending on what is specified in the config for the model:
        1. If a `model` is provided, run selfchat for model
        2. If a `log_path` is provided, simply load the log path
        3. If a `task` is provided, convert the task to ACUTE format and load that.
        """
        for model in self.models:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._print_progress(f'Running self-chat for {model}')
            outfile = self._get_log_path(model)

            if not os.path.exists(outfile):
                if 'model' in self.model_config[model]:
                    config = self._get_selfchat_config(model)
                    with capture_output():
                        parser = self_chat_setup_args()
                        parser.set_params(**config)
                        opt = parser.parse_args(args=[])
                    self_chat(opt)
                elif 'task' in self.model_config[model]:
                    self._convert_task_to_conversations(model)
                else:
                    raise RuntimeError(
                        f'Path must exist if log_path specified for {model}'
                    )

                if os.path.exists(outfile):
                    self._print_progress(f'Chats saved to {outfile} for {model}')

            self._print_progress(f'Chats already exist in {outfile}, moving on...')
            self.chat_files[model] = outfile

    def run_acute_eval(self):
        """
        Run ACUTE Eval.
        """
        self.set_up_acute_eval()
        db, cfg = load_db_and_process_config(self.args)
        print(f'*** RUN ID: {cfg.mephisto.task.task_name} ***')
        operator = Operator(db)
        operator.validate_and_run_config(run_config=cfg.mephisto, shared_state=None)
        operator.wait_for_runs_then_shutdown(
            skip_input=True, log_rate=cfg.monitoring_log_rate
        )

    def set_up_acute_eval(self):
        self._load_pairings_file()

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

    def analyze_results(self, args: Optional[str] = None):
        """
        Analyze results of ACUTE Eval run, using the optional input args.

        Save results to appropriate filepath.
        """
        self._print_progress(f'Analyzing Results for run id {self.run_id}')
        parser = analysis_setup_args()
        if args is not None:
            arg_string = args.split()
        else:
            arg_string = []
        opt = parser.parse_args(arg_string)
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
                'run_ids': self.run_id,
                'root_dir': self.fast_acute_args.root_dir,
                'outdir': self.results_path,
                'task': self.task,
            }
        )

        analyzer = self.ANALYZER(opt)
        self.results = analyzer.get_matchup_totals_with_significance()
        analyzer.save_results()

        self._print_progress(f'ACUTE Results: {self.results}')
        self._print_progress(f'ACUTE results saved to {self.results_path}')


defaults = [
    {"mephisto/blueprint": FAST_ACUTE_BLUEPRINT_TYPE},
    {"mephisto/architect": "local"},
    {"mephisto/provider": "mock"},
    'conf/base_fast_acute',
    {"conf": "example_fast_acute"},
]


@dataclass
class TestScriptConfig(MTurkRunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY
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
    runner.compile_chat_logs()

    # Run ACUTE-Eval
    runner.run_acute_eval()

    # Analyze the results
    runner.analyze_results()


if __name__ == '__main__':
    main()
