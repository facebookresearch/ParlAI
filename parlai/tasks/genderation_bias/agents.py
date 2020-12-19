#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generates a controllable_gen version of a ParlAI task, i.e., for tasks with multi-turn
dialogues (episodes), this will generate a task with single example episodes, in which
we append to the context a special classification token.

In order to use this teacher, specify the task flag as follows:
`--task genderation_bias:controllable_task:<ORIGINAL TASK NAME>`.

As an example, try running:

`parlai display_data -t genderation_bias:controllable_task:convai2`
"""

from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher
from parlai.utils.io import PathManager
import parlai.utils.logging as logging
from parlai.utils.typing import TShared

from parlai.tasks.genderation_bias.build import build
from parlai.tasks.genderation_bias.utils import (
    flatten_and_classify,
    get_original_task_module,
)

from copy import deepcopy
import datetime
import glob
import json
import os
from tqdm import tqdm
from typing import List, Optional, Tuple


class ControllableTaskTeacher(FixedDialogTeacher):
    """
    Generates a controllable_gen version of a ParlAI task, i.e., for tasks with multi-
    turn dialogues (episodes), this will generate a task with single example episodes,
    in which we append to the context a special classification token.
    """

    @staticmethod
    def add_cmdline_args(parser):
        flattened = parser.add_argument_group('ControllableTaskTeacher Flattening Args')
        flattened.add_argument(
            '--flatten-include-labels',
            type='bool',
            default=True,
            help='Include labels in the history when flattening an episode',
        )
        flattened.add_argument(
            '--flatten-delimiter',
            type=str,
            default='\n',
            help='How to join the dialogue history from previous turns.',
        )
        flattened.add_argument(
            '--flatten-max-context-length',
            type=int,
            default=-1,
            help='Maximum number of utterances to include per episode. '
            'Default -1 keeps all.',
        )
        agent = parser.add_argument_group('ControllableTaskTeacher Args')
        agent.add_argument(
            '--invalidate-cache',
            type='bool',
            default=False,
            help='Set this to True to rebuild the data (may want to do this if '
            'original data has changed or you want to rebuild with new options)',
        )
        agent.add_argument(
            '--max-examples',
            type=int,
            default=-1,
            help='If greater than zero, will stop building after a certain num of exs',
        )
        agent.add_argument(
            '--fixed-control',
            type=str,
            default='',
            help='Always append this fixed control string, good for deploy time.',
        )
        # Add the arguments for the task teacher
        opt = parser.parse_and_process_known_args()[0]
        tasks = get_original_task_module(opt, multi_possible=True)
        for task in tasks:
            if hasattr(task, 'add_cmdline_args'):
                task.add_cmdline_args(parser)

        return parser

    def __init__(self, opt: Opt, shared: TShared = None):
        assert opt['flatten_delimiter'] == opt.get(
            'delimiter', '\n'
        ), '--flatten-delimiter and --delimiter are set differently, please inspect and set to the same to avoid unexpected results'
        self.opt = opt

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self.word_lists = self.build_wordlists(opt)
            self.data = self._setup_data(opt)

        super().__init__(opt, shared)
        self.reset()

    def num_episodes(self) -> int:
        return len(self.data)

    def num_examples(self) -> int:
        return len(self.data)

    def _get_save_path(self, datapath: str, date: str) -> str:
        """
        Return save path for the controllable gen data.

        :param datapath:
            path to ParlAI Data
        :param date:
            current date

        :return path:
            return path to save
        """
        return os.path.join(
            datapath,
            f"{self.original_task_name.replace(':', '_')}_flattened_controllable_gen_{date}",
        )

    @classmethod
    def build_wordlists(cls, opt: Opt) -> Tuple[List[str], List[str]]:
        """
        Load list of explicitly gendered words.

        Words taken from <https://github.com/uclanlp/gn_glove/blob/master/wordlist/>.

        Examples include brother, girl, actress, husbands, etc.
        """
        build(opt['datapath'])
        folder = os.path.join(opt['datapath'], 'genderation_bias')
        male_words = os.path.join(folder, 'male_word_file.txt')
        female_words = os.path.join(folder, 'female_word_file.txt')

        with open(male_words, 'r') as f:
            male = f.read().splitlines()

        with open(female_words, 'r') as f:
            female = f.read().splitlines()

        return male, female

    def _setup_data(self, opt: Opt) -> List[List[Message]]:
        """
        Flatten and classify the normal task data.

        Save/load where applicable.

        :param opt:
            options dict.
        """
        # create save directory, if it does not already exist
        self.original_task_name = ':'.join(opt['task'].split(':')[2:])
        self.save_dir = self._get_save_path(
            opt['datapath'], str(datetime.datetime.today())
        )
        os.makedirs(self.save_dir, exist_ok=True)

        fname = f"{opt['datatype'].split(':')[0]}.json"
        self.save_path = os.path.join(self.save_dir, fname)

        data = self.load_data(opt, fname)
        if data is not None:
            # successfully load data
            return data

        # build the original teacher
        original_task_module = get_original_task_module(opt)
        teacher_opt = deepcopy(opt)
        teacher_opt['task'] = self.original_task_name
        teacher = original_task_module(teacher_opt)

        total_exs = teacher.num_examples()
        if self.opt['max_examples'] > 0:
            total_exs = min(self.opt['max_examples'], total_exs)

        progress_bar = tqdm(
            total=total_exs, unit='ex', unit_scale=True, desc='Building flattened data'
        )

        all_episodes = []
        num_exs = 0
        while num_exs < total_exs:
            current_episode = []
            episode_done = False

            while not episode_done:
                action = Message(teacher.act())
                current_episode.append(action)
                episode_done = action.get('episode_done', False)
                num_exs += 1

            # flatten the episode into 1-example episodes with context
            flattened_ep = flatten_and_classify(
                current_episode,
                opt['flatten_max_context_length'],
                include_labels=opt['flatten_include_labels'],
                delimiter=opt['flatten_delimiter'],
                word_lists=self.word_lists,
            )
            all_episodes += flattened_ep

            progress_bar.update(len(flattened_ep))

        # save data for future use
        self.save_data(all_episodes)

        return all_episodes

    def load_data(self, opt: Opt, filename: str) -> Optional[List[List[Message]]]:
        """
        Attempt to load pre-build data.

        Checks for the most recently build data via the date string.

        :param opt:
            options dict
        :param filename:
            name of (potentially) saved data

        :return episodes:
            return list of episodes, if available
        """
        # first check for the most recent date
        save_dir = self._get_save_path(opt['datapath'], '*')
        all_dates = []
        for fname in glob.glob(os.path.join(save_dir, filename)):
            date = os.path.split(fname)[0].split('_')[-1]
            all_dates.append(date)

        if len(all_dates) > 0:
            most_recent = os.path.join(
                self._get_save_path(opt['datapath'], sorted(all_dates)[-1]), filename
            )
        else:
            # data has not been built yet
            return None

        if opt['invalidate_cache']:
            # invalidate the cache and remove the existing data
            logging.warn(
                f' [ WARNING: invalidating cache at {self.save_path} and rebuilding the data. ]'
            )
            if self.save_path == most_recent:
                os.remove(self.save_path)
            return None

        # Loading from most recent date
        self.save_path = most_recent
        logging.info(f' [ Data already exists. Loading from: {self.save_path} ]')
        with PathManager.open(self.save_path, 'rb') as f:
            data = json.load(f)

        return data

    def save_data(self, data: List[List[Message]]):
        """
        Save the data via dumping to a json file.

        :param data:
            list of episodes
        """
        try:
            json_data = json.dumps(data)
            with PathManager.open(self.save_path, 'w') as f:
                f.write(json_data)
            logging.info(f'[ Data successfully saved to path: {self.save_path} ]')
        except Exception:
            logging.warn('Data is not json serializable; not saving')

    def get(self, episode_idx: int, entry_idx: int = 0) -> Message:
        """
        Return a flattened example.

        If using a fixed control, put that in instead of what was originally in the text.

        :param episode_idx:
            index of ep in data
        :param entry_idx:
            index of ex in ep

        :return ex:
            return an example
        """
        ex = Message(self.data[episode_idx])

        if self.opt['fixed_control'] != '':
            old_text = ' '.join(ex['text'].split(' ')[:-1])
            text = f"{old_text} {self.opt['fixed_control']}"
            ex.force_set('text', text)

        return ex

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class DefaultTeacher(ControllableTaskTeacher):
    pass
