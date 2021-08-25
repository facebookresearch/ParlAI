#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from parlai.crowdsourcing.utils.acceptability import AcceptabilityChecker
from parlai.crowdsourcing.utils.analysis import AbstractTurnAnnotationResultsCompiler
from parlai.crowdsourcing.tasks.model_chat.analysis.render_html import render_live_mturk

TASK_DIR = '/private/home/jingxu23/ParlAI/parlai/crowdsourcing/tasks/model_chat'

MODEL_NICKNAMES = [
    'BST2.7B',
    'MSC2.7B_128',
    'MSC2.7B_1024',
    'RAG_SUMMSC',
    'FiD_SUMMSC',
    'FiDRAG_SUMMSC',
]

OVERRIDE_OPT_FILENAMES = {
    'MSC2.7B_128': 'override_opt_msc128.json',
    'MSC2.7B_1024': 'override_opt_msc1024.json',
    'BST2.7B': 'override_opt_bb.json',
    'RAG_SUMMSC': 'override_opt_ragsummsc.json',
    'FiD_SUMMSC': 'override_opt_fidsummsc.json',
    'FiDRAG_SUMMSC': 'override_opt_fidragsummsc.json',
    'RAG_RAWHISTORY_NDOC10': 'override_opt_ragrawhistoryndoc10.json',
}


def get_output_folder(model_nickname, is_engaging):
    subfolder = 'engaging' if is_engaging else 'normal'
    return os.path.join(TASK_DIR, f'results/{subfolder}/{model_nickname}')


def get_results_folder(model_nickname, is_engaging):
    subfolder = 'engaging' if is_engaging else 'normal'
    return [os.path.join(TASK_DIR, f'model_chat/{subfolder}/{model_nickname}')]


import shlex
import subprocess


def bash(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = str(output)
    output = output[:-3]
    output = output.lstrip('b').strip('\'').strip('"')
    return output


class ModelChatResultsCompiler(AbstractTurnAnnotationResultsCompiler):
    """
    Compile and save results of human+model chats.

    Results will be saved on the level of specific conversations, as well as aggregated
    up the level of each worker as a whole.
    """

    @classmethod
    def setup_args(cls):
        parser = super().setup_args()
        parser.add_argument(
            '--start-date',
            type=str,
            default='',
            help='The earliest date to analyze results from',
        )
        parser.add_argument(
            '--model-nickname', type=str, default='', help='name of the model'
        )
        parser.add_argument(
            '--is-engaging', type=bool, default=False, help='whether if it is engaging'
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):

        super().__init__(opt)

        # Input args
        self.model_nickname = opt['model_nickname']
        self.is_engaging = opt['is_engaging']
        self.results_folders = get_results_folder(self.model_nickname, self.is_engaging)
        assert len(self.results_folders) > 0
        for folder in self.results_folders:
            assert os.path.isdir(folder), f'{folder} is not a valid folder!'
        self.output_folder = get_output_folder(self.model_nickname, self.is_engaging)
        os.makedirs(self.output_folder, exist_ok=True)
        self.start_date = opt['start_date']

        # Setting up problem buckets
        self.regular_buckets = [
            bucket
            for bucket in self.problem_buckets
            if bucket not in ['other', 'none_all_good']
        ]
        # Remove the buckets that are special cases

        self.acceptability_checker = AcceptabilityChecker()

    def get_results_path_base(self) -> str:
        now = datetime.now()
        return os.path.join(
            self.output_folder, f'results_{now.strftime("%Y%m%d_%H%M%S")}'
        )

    def compile_results(self) -> pd.DataFrame:
        """
        python /private/home/jingxu23/ParlAI/parlai/crowdsourcing/tasks/model_chat/analy
        sis/render.py \

        --problem-buckets they,you,new,none,engaging \
        --is-engaging True \
        --model-nickname KNOWLEDGE_BOT_MEM
        FiD_SUMMSC
        """

        read_folders = []
        date_strings = []
        for folder in self.results_folders:
            # Load paths
            date_strings = sorted(
                [
                    obj
                    for obj in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, obj))
                    and re.fullmatch(r'\d\d\d\d_\d\d_\d\d', obj)
                ]
            )
            if self.start_date != '':
                date_strings = [
                    str_ for str_ in date_strings if str_ >= self.start_date
                ]
            folders = [os.path.join(folder, str_) for str_ in date_strings]
            read_folders.extend(folders)
        print(f'Date folders: ' + ', '.join(date_strings))

        now = datetime.now()
        render_json_file = os.path.join(
            self.output_folder,
            f'render_{self.model_nickname}_{now.strftime("%Y%m%d")}.json',
        )
        render_html_file = os.path.join(
            self.output_folder,
            f'render_{self.model_nickname}_{now.strftime("%Y%m%d")}.html',
        )
        # Read in each file
        num_incomplete_convos = 0
        num_complete_convos = 0
        all_data = []
        for read_folder in read_folders:
            for file_name in sorted(os.listdir(read_folder)):
                if 'sandbox' in file_name:
                    continue

                if 'incomplete' in file_name:
                    num_incomplete_convos += 1
                    continue
                else:
                    num_complete_convos += 1

                # Read in file
                with open(os.path.join(read_folder, file_name), 'rb') as f:
                    data = json.load(f)
                context_data = [
                    {
                        'text': "bot's persona: (speaker2)"
                        + " ".join(data['bot_persona_strings'])
                        + "\n"
                        + "human's persona: (speaker1)"
                        + " ".join(data['human_persona_strings'])
                        + "\n"
                        # + "_____bot observe_____: &#10" + data['context_info']['observation_for_bot']['text']
                        # + "&#10"
                        # + "&#10"
                        + "[initial_data_id]: "
                        + data['context_info']['observation_for_bot'][
                            'initial_data_id'
                        ],
                        'id': 'context',
                        'episode_done': False,
                    },
                    {
                        'text': "bot persona: (Blue)"
                        + " ".join(data['context_info']['your_persona_strings'])
                        + "&#10"
                        + "human persona: (Grey)"
                        + " ".join(data['context_info']['their_persona_strings']),
                        'id': 'context',
                        'episode_done': False,
                    },
                    {
                        'text': "[context from previous 4 sessions]",
                        'id': data['dialog'][0]['id'],
                        'episode_done': False,
                    },
                ]
                clean_dialog = [
                    {
                        'text': turn['text'],
                        'id': turn['id'],
                        'episode_done': turn.get('episode_done', False),
                    }
                    for turn in data['dialog']
                ]
                data['dialog'] = context_data + clean_dialog
                all_data.append(data)
        with open(render_json_file, 'w') as fw:
            for data in all_data:
                fw.write(json.dumps(data) + '\n')
        bashCommand = f'python /private/home/jingxu23/ParlAI/parlai/crowdsourcing/tasks/model_chat/analysis/render_html.py --logs-path {render_json_file} --html-path {render_html_file}'
        import subprocess

        p = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        result = out.decode()
        print(result)


if __name__ == '__main__':
    parser_ = ModelChatResultsCompiler.setup_args()
    args_ = parser_.parse_args()
    ModelChatResultsCompiler(vars(args_)).compile_results()
