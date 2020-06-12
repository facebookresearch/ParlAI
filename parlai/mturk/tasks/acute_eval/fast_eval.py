#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai import __file__ as parlai_filepath
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.acute_eval.run import AcuteEvaluator, add_args as acute_add_args
from parlai.scripts.self_chat import self_chat, setup_args as self_chat_setup_args
from parlai.utils.conversations import Conversations, Conversation
from parlai.utils.strings import normalize_reply
from parlai.utils.testing import capture_output

from parlai.mturk.tasks.acute_eval.analysis import (
    AcuteAnalyzer,
    setup_args as analysis_setup_args,
)
from parlai.mturk.tasks.acute_eval.dump_task_to_acute_format import (
    dump_data as convert_task_data,
    setup_args as convert_task_setup_args,
)
from parlai.mturk.tasks.acute_eval.configs import CONFIG

try:
    from parlai_internal.projects.fast_acute.model_configs import (
        CONFIG as internal_conf,
    )

    CONFIG.update(internal_conf)
except ImportError:
    # No access to internal
    pass

from typing import Dict, Any, List, Tuple, Set
from itertools import combinations

import datetime
import time
import json
import os
import random
import torch
import hashlib

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
}
EXAMPLE_PATH = os.path.join(
    os.path.dirname(parlai_filepath), 'mturk/tasks/acute_eval/example'
)
# Feel free to edit this, but not necessary
SUBTASKS_PER_HIT = 5
MAX_HITS_PER_WORKER = 1
MATCHUPS_PER_PAIR = 160

ACUTE_DEFAULT_ARGS = {
    # onboarding
    'block-on-onboarding-fail': True,
    # pairings
    # general mturk
    'reward': 0.5,
    'max_hits_per_worker': MAX_HITS_PER_WORKER,
    'assignment_duration_in_seconds': 600,
    'auto_approve_delay': 5,
    # acute args
    'annotations_per_pair': 1,
    'seed': 42,
    'subtasks_per_hit': SUBTASKS_PER_HIT,
    # Task Config
    'task_config': {
        'hit_title': 'Which Conversational Partner is Better?',
        'hit_description': 'Evaluate quality of conversations through comparison.',
        'hit_keywords': 'chat,evaluation,comparison,conversation',
    },
    # temp directory for MTURK
    'tmp_dir': '/tmp',
}

#######################
# SELF CHAT CONSTANTS #
#######################
NUM_SELFCHAT_EXAMPLES = 100
SELFCHAT_MAX_TURNS = 6

##############################
# TASK CONVERSTION CONSTANTS #
##############################
NUM_TASK_DATA_EPISODES = 500
SELFCHAT_MAX_TURNS = 6


def setup_args(parser=None) -> ParlaiParser:
    """
    Setup args.
    """
    parser = ParlaiParser(True, False)
    parser.add_argument(
        '--ids',
        type='nonestr',
        help='Comma separated list of CONFIG ids for round robin evaluation (must be at least 2)',
        default=None,
    )
    parser.add_argument(
        '--id-pairs',
        type='nonestr',
        help='Comma separated, colon-delimited list of CONFIG pairs for evaluation, '
        'e.g. model1:model2,model1:model3',
        default=None,
    )
    parser.add_argument(
        '-eval',
        '--acute-eval-type',
        type=str,
        default='engaging',
        choices=list(ACUTE_EVAL_TYPES.keys()),
        help='which evaluation to run for acute',
    )
    parser.add_argument(
        '-mpp',
        '--matchups-per-pair',
        type=int,
        default=MATCHUPS_PER_PAIR,
        help='How many matchups to generate for each pair of ids.',
    )
    parser.add_argument(
        '--live-acute',
        type='bool',
        default=False,
        help='whether this is a LIVE acute run. ',
    )
    parser.add_argument(
        '--onboarding-path',
        type=str,
        default=os.path.join(EXAMPLE_PATH, 'onboarding.jsonl'),
        help='path to onboarding pair',
    )
    parser.set_defaults(selfchat_task=True, task='self_chat')
    return parser


class ParlAIQuickAcute(object):
    """
    Execute Quick ACUTE Runs.
    """

    def __init__(self, opt: Opt):
        self.opt: Opt = opt
        # ids + task
        self._build_id_pairs()
        self.task: str = opt['task']

        # keep track of chat files per model
        self.chat_files: Dict[str, str] = {}

        # question config for running ACUTE
        self.question_config: Dict[str, str] = ACUTE_EVAL_TYPES[opt['acute_eval_type']]

        # root directory for saving everything
        self.root_dir = os.path.join(opt['datapath'], 'acute_eval')
        os.makedirs(self.root_dir, exist_ok=True)

        # path to onboarding file
        self.onboarding_path = opt['onboarding_path']

    #############
    # Utilities #
    #############
    def _build_id_pairs(self):
        """
        Generate self.config_ids and self.combos from self.opt.
        """
        choices = CONFIG.keys()
        combos: Set[Tuple[str, str]] = set()
        ids: Set[str] = set()
        if self.opt['ids'] is None and self.opt['id_pairs'] is None:
            raise RuntimeError(
                'Either --ids or --id-pairs should be set for comparision.'
            )
        if self.opt['id_pairs'] is not None:
            id_pairs = self.opt['id_pairs'].split(',')
            id_pairs = [id_pair.split(':') for id_pair in id_pairs]
            for id_pair in id_pairs:
                combos.add(tuple(sorted((id_pair[0], id_pair[1]))))
                ids |= set(id_pair)
        else:
            ids = set(self.opt['ids'].split(','))
            combos = set(combinations(ids, 2))
        self.config_ids: List[str] = list(ids)
        self.config_ids.sort()
        self.combos: List[Tuple[str, str]] = list(combos)
        self.combos.sort()
        # verify that ids are contained in the config:
        for config_id in self.config_ids:
            if config_id not in choices:
                raise RuntimeError(
                    f'ID {config_id} not specified in the config (`configs.py`).'
                )
        assert len(self.config_ids) > 1, 'Must specify least 2 ids'

    def _print_progress(self, msg: str):
        """
        Format a msg to print to stdout well.

        :param msg:
            message to print
        """
        print(f"\n{'-' * 60}\n {msg} \n {'-' * 60}")

    def _get_selfchat_config(self, config_id: str) -> Dict[str, Any]:
        """
        Return config for selfchat.

        :param config_id:
            config_id string

        :return config:
            dict config for self-chat
        """
        outfile = self._get_selfchat_log_path(config_id)
        config = CONFIG[config_id]
        config.update(
            {
                'task': self.task,
                'outfile': outfile,
                'num_self_chats': NUM_SELFCHAT_EXAMPLES,
                'selfchat_max_turns': SELFCHAT_MAX_TURNS,
                'display_examples': False,
                'log_every_n_secs': -1,
                'indent': -1,
            }
        )
        return config

    def _get_task_conversion_config(self, config_id: str) -> Dict[str, Any]:
        """
        Return config for task conversion to conversation format.

        :param config_id:
            config_id string

        :return config:
            dict config for task conversion script
        """
        outfile = self._get_task_data_path(config_id)
        config = CONFIG[config_id]
        config.update(
            {
                'outfile': outfile,
                'num_episodes': NUM_TASK_DATA_EPISODES,
                'speaker_0_id': f'{config_id}_as_human',
                'speaker_1_id': config_id,
            }
        )
        return config

    def _get_vs_path(self, subdir: str) -> str:
        """
        Return a unique path for the set of comparison combos given a subdirectory.

        We hash the filename as it can grow quite large.
        """
        assert subdir
        os.makedirs(os.path.join(self.root_dir, subdir), exist_ok=True)

        def _combo_name(id1, id2):
            """
            Return joined name for combo of comparisons.
            """
            id1_name = id1
            id2_name = id2
            if 'model' in CONFIG[id1]:
                id1_name += self.task.replace(':', '_')
            if 'model' in CONFIG[id2]:
                id2_name += self.task.replace(':', '_')
            return f'{id1_name}__vs__{id2_name}'

        return os.path.join(
            self.root_dir,
            subdir,
            hashlib.sha1(
                '___and___'.join(
                    [f"{_combo_name(id1, id2)}" for id1, id2 in self.combos]
                ).encode('utf-8')
            ).hexdigest()[:10],
        )

    def _get_log_path(self, config_id: str) -> str:
        """
        Return path to chat logs given config_id.

        :param identifier:
            config_id in CONFIG.
        """
        config = CONFIG[config_id]
        path = ''
        if 'log_path' in config:
            path = config['log_path']
            assert os.path.exists(
                path
            ), f'Path provided in log_path for {config_id} does not exist'
        elif 'task' in config:
            path = self._get_task_data_path(config_id)
        elif 'model' in config:
            path = self._get_selfchat_log_path(config_id)

        assert path, f'Invalid config for {config_id}'

        return path

    def _get_task_data_path(self, config_id: str) -> str:
        """
        Return path to task data as conversations for given task.

        :param config_id:
            config_id string
        """
        task_data_dir = os.path.join(self.root_dir, 'tasks_as_conversations')
        os.makedirs(task_data_dir, exist_ok=True)
        return os.path.join(task_data_dir, f"{config_id}.jsonl")

    def _get_selfchat_log_path(self, config_id: str) -> str:
        """
        Return path to selfchat log for a given model.

        :param config_id:
            config_id string
        """
        self_chat_dir = os.path.join(self.root_dir, 'self_chats')
        os.makedirs(self_chat_dir, exist_ok=True)
        return os.path.join(
            self_chat_dir, f"{config_id}.{self.task.replace(':', '_')}.jsonl"
        )

    def _acutify_convo(
        self, conversation: Conversation, config_id: str
    ) -> Dict[str, List]:
        """
        Format world-logged conversation to be ACUTE format.

        :param conversation:
            dictionary containing the dialogue for a config_id
        :param config_id:
            config_id string

        :return conversation:
            An ACUTE-Readable conversation
        """
        config = CONFIG[config_id]
        is_selfchat = 'model' in config
        acute_conversation: Dict[str, List] = {
            'context': [],
            'dialogue': [],
            'speakers': [],
        }
        for i, ex in enumerate(conversation):
            if ex['id'] == 'context':
                acute_conversation['context'].append(ex)
                continue
            speaker_id = ex['id']
            if is_selfchat:
                speaker_id = config_id if i % 2 == 0 else f'other_{config_id}'
            if speaker_id not in acute_conversation['speakers']:
                acute_conversation['speakers'].append(speaker_id)
            acute_conversation['dialogue'].append(
                {'id': speaker_id, 'text': normalize_reply(ex['text'])}
            )
        return acute_conversation

    def _get_unique_ids(
        self, conversations: Dict[str, Conversations]
    ) -> Dict[str, List[int]]:
        """
        Assign unique IDs for each conversation in conversations.

        This is important for ACUTE-Eval, since we do not want evaluators
        to see the same conversations across comparisons.

        :param conversations:
            Dict mapping config ID to list of conversations

        :return unique_ids:
            dict mapping config id to list of conversation IDs
        """
        id_num = 0
        unique_ids: Dict[str, List[int]] = {}
        for config_id in self.config_ids:
            unique_ids[config_id] = []
            for _convo in conversations[config_id]:
                unique_ids[config_id].append(id_num)
                id_num += 1

        return unique_ids

    def _build_conversation_pairs(
        self, conversations: Dict[str, Conversations]
    ) -> List[Dict[str, Any]]:
        """
        Build a conversation pair to show during ACUTE Eval.

        We build twice as many pairs per matchup as specified
        in the config, to account for issues where sometimes
        we run out of pairs of conversations to evaluate.

        :param conversations:
            A dictionary mapping config_id to dialogues

        :return pairs:
            A list of conversation pairs
        """
        unique_ids = self._get_unique_ids(conversations)
        pairs = []
        pairs_per_id = self.opt['matchups_per_pair'] * 2
        # Write random pairs of conversations
        for id_pair in self.combos:
            for _ in range(pairs_per_id):
                conversation_indices = [
                    random.choice(range(len(conversations[id_]))) for id_ in id_pair
                ]
                pair = []
                pair_ids = []
                for i, c_id in enumerate(conversation_indices):
                    id_ = id_pair[i]
                    pair.append(self._acutify_convo(conversations[id_][c_id], id_))
                    pair_ids.append(unique_ids[id_][c_id])
                pairs.append(
                    {
                        "is_onboarding": False,
                        "speakers_to_eval": id_pair,
                        "dialogue_dicts": pair,
                        "dialogue_ids": pair_ids,
                    }
                )
        return pairs

    def _build_pairings_file(self):
        """
        Build and save pairings to pairings file.
        """
        onboarding_pairs = []
        with open(self.onboarding_path) as f:
            for line in f:
                onboarding_pairs.append(json.loads(line))

        pairings_filepath = self._get_vs_path('pairings_files')

        self._print_progress(f'building pairings file, saving at {pairings_filepath}')
        conversations = {
            config_id: Conversations(self.chat_files[config_id])
            for config_id in self.config_ids
        }
        pairs = self._build_conversation_pairs(conversations)

        with open(pairings_filepath, 'w') as f:
            # Write the onboarding convo
            pairs = onboarding_pairs + pairs
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

    def _load_pairings_file(self):
        """
        Build the pairings file for ACUTE-Eval.

        If a pairings file already exists, we ask the user whether they would like to
        overwrite the pairings file.
        """
        pairings_filepath = self._get_vs_path('pairings_files')
        # Rebuild pairings file if necessary
        if not os.path.exists(pairings_filepath):
            self._build_pairings_file()
        else:
            modify_time = os.path.getmtime(pairings_filepath)
            self._print_progress(
                f'Pairings already exist {pairings_filepath}. Last modified {time.ctime(modify_time)}'
            )
            answer = ''
            while answer.lower().strip() != 'y' and answer.lower().strip() != 'o':
                answer = input('Enter y to use, o to overwrite:')
                if answer.lower().strip() == 'o':
                    self._build_pairings_file()

        # Load the file
        self._print_progress(f'loading pairings file from {pairings_filepath}')
        self.pairings_filepath = pairings_filepath

    def _run_selfchat(self, config_id: str):
        """
        Run self-chat for model.

        :param config_id:
            id in config
        """
        self._print_progress(f'Running self-chat for {config_id}')
        config = self._get_selfchat_config(config_id)

        with capture_output():
            parser = self_chat_setup_args()
            parser.set_params(**config)
            opt = parser.parse_args(args=[], print_args=False)
        self_chat(opt)

    def _convert_task_to_conversations(self, config_id: str):
        """
        Convert task data to conversations format.

        :param config_id:
            id in config
        """
        self._print_progress(
            f'Converting task data to conversations format for {config_id}'
        )
        config = self._get_task_conversion_config(config_id)

        with capture_output():
            parser = convert_task_setup_args()
            parser.set_params(**config)
            opt = parser.parse_args(args=[], print_args=False)
        convert_task_data(opt)

    ##################
    # Main Functions #
    ##################
    def compile_chat_logs(self):
        """
        Compile Chat Logs.

        Logs are generated depending on what is specified in the config for the model:
        1. If a `model` is provided, run selfchat for model
        2. If a `log_path` is provided, simply load the log path
        3. If a `task` is provided, convert the task to ACUTE format and load that.
        """
        for model in self.config_ids:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            outfile = self._get_log_path(model)

            if not os.path.exists(outfile):
                # 1. Self-chat; 2. Task
                if 'model' in CONFIG[model]:
                    self._run_selfchat(model)
                elif 'task' in CONFIG[model]:
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
        self._load_pairings_file()

        self.acute_args = acute_add_args(print_args=False)
        self.acute_args.update(ACUTE_DEFAULT_ARGS)
        total_convos = self.opt['matchups_per_pair'] * len(self.combos)
        self.acute_args.update(
            {
                'is_sandbox': not self.opt['live_acute'],
                'pairings_filepath': self.pairings_filepath,
                's1_choice': self.question_config['s1_choice'],
                's2_choice': self.question_config['s2_choice'],
                'question': self.question_config['question'],
                'num_matchup_pairs': total_convos,
                'num_conversations': int(
                    total_convos / (SUBTASKS_PER_HIT - 1)  # subtract 1 for onboarding
                ),
            }
        )
        self.acute_evaluator = AcuteEvaluator(self.acute_args)
        if self.opt['live_acute']:
            self._print_progress('Running ACUTE-EVAL in LIVE Mode')
        else:
            self._print_progress('Running ACUTE-EVAL in SANDBOX Mode')
        self.run_id = self.acute_evaluator.run()

    def analyze_results(self):
        """
        Analyze results of ACUTE Eval run.

        Save results to appropriate filepath.
        """
        self._print_progress(f'Analyzing Results for run id {self.run_id}')
        parser = analysis_setup_args()
        opt = parser.parse_args([], print_args=False)
        today = datetime.date.today().isoformat()
        self.results_path = f"{self._get_vs_path(f'acute_results/{today}/')}"
        opt.update(
            {
                'run_id': self.run_id,
                'outdir': self.results_path,
                'pairings_filepath': self.pairings_filepath,
            }
        )
        opt.update(self.acute_args)

        analyzer = AcuteAnalyzer(opt, self.run_id)
        self.results = analyzer.get_matchup_totals_with_signficance()
        analyzer.save_results()

        self._print_progress(f'ACUTE Results: {self.results}')
        self._print_progress(f'ACUTE results saved to {self.results_path}')


if __name__ == '__main__':
    parser = setup_args()
    runner = ParlAIQuickAcute(parser.parse_args(print_args=False))

    # Compile Chat Logs
    runner.compile_chat_logs()

    # Run ACUTE Eval
    runner.run_acute_eval()

    # Analyze the Results
    runner.analyze_results()
