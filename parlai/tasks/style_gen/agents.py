#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import List, Optional

from tqdm import tqdm

from parlai.core.opt import Opt
from parlai.tasks.convai2.agents import DefaultTeacher as Convai2DefaultTeacher
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.style_gen.build import (
    build_personality_list,
    build_style_labeled_datasets,
    TASK_FOLDER_NAME,
)
from parlai.tasks.wizard_of_wikipedia.agents import WizardDialogKnowledgeTeacher
from parlai.utils.misc import warn_once


def get_style_labeled_data_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build_style_labeled_datasets(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], TASK_FOLDER_NAME, opt['task'], dt + '.txt')


def get_personality_list_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build_personality_list(opt)
    return os.path.join(opt['datapath'], TASK_FOLDER_NAME, 'personality_list.txt')


# TODO: revise below
class ConvAI2PersonaTopicifierTeacher(Convai2DefaultTeacher):
    """
    Adds WoW topics to ConvAI2 data.
    """

    def __init__(self, opt, shared=None):
        if 'stream' in opt['datatype']:
            warn_once('Warning: this teacher is not compatible with StreamDialogData!')
            # StreamDialogData works by reading directly from a text file without any
            # alteration, but this teacher must append a WoW topic string to the context
            # of the first example of each episode.
            assert opt['datatype'].endswith(':stream')
            opt['datatype'] = opt['datatype'][: -len(':stream')]
        self.persona_topicifier = PersonaTopicifier(
            opt=opt, should_have_personas=True, should_have_topics=False
        )
        super().__init__(opt, shared=shared)

    def get(self, episode_idx, entry_idx=None):
        gotten = super().get(episode_idx, entry_idx=entry_idx)
        if entry_idx == 0:
            modified_text = self.persona_topicifier.get_modified_text(gotten['text'])
            gotten.force_set('text', modified_text)
        return gotten


class WoWPersonaTopicifierTeacher(WizardDialogKnowledgeTeacher):
    """
    Adds personas to WoW data.
    """

    def __init__(self, opt, shared=None):
        self.persona_topicifier = PersonaTopicifier(
            opt=opt, should_have_personas=False, should_have_topics=True
        )
        super().__init__(opt, shared=shared)

    def get(self, episode_idx, entry_idx=None):
        gotten = super().get(episode_idx, entry_idx=entry_idx)
        if entry_idx == 0:
            modified_text = self.persona_topicifier.get_modified_text(gotten['text'])
            gotten['text'] = modified_text
        return gotten


class EDPersonaTopicifierTeacher(EmpatheticDialoguesTeacher):
    """
    Adds persona and WoW topic to ED context strings.
    """

    RECOMPILE_DEFAULT = False

    @classmethod
    def add_cmdline_args(cls, argparser):
        EmpatheticDialoguesTeacher.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('EDPersonaTopicifierTeacher arguments')
        agent.add_argument(
            '--recompile-persona-topic-data',
            type='bool',
            default=cls.RECOMPILE_DEFAULT,
            help='Re-compile data with ConvAI2 personas and WoW topics added. Only useful for demonstrating how data was produced.',
        )

    def __init__(self, opt, shared=None):
        self.persona_topicifier = PersonaTopicifier(
            opt=opt, should_have_personas=False, should_have_topics=False
        )
        super().__init__(opt, shared=shared)

        if (
            self.remove_political_convos is True
            or self.opt.get('deepmoji') is not None
            or self.opt.get('fasttextloc') is not None
            or self.opt.get('prepend', -1) > 0
        ):
            raise NotImplementedError(
                'Removing political conversations or using deepmoji, fasttextloc, or '
                'prepend not supported with this teacher.'
            )

        # Running over all examples is really slow because the process of finding a WoW
        # topic is expensive, so let's load cached data with personas and topics unless
        # --recompile-persona-topic-data is True
        if opt.get('recompile_persona_topic_data', self.RECOMPILE_DEFAULT):
            self.data_path = (
                _cached_data_path(
                    opt=self.opt, experiencer_side_only=self.experiencer_side_only
                )
                + '.recompiled'
            )
            warn_once(f'Compiling data file for {self.data_path}.')
            self.persona_topic_data = self._compile_data()
            warn_once(f'Saving data to {self.data_path}.')
            with open(self.data_path, 'w') as f_write:
                json.dump(self.persona_topic_data, f_write)
        else:
            self.data_path = _cached_data_path(
                opt=self.opt, experiencer_side_only=self.experiencer_side_only
            )
            warn_once(f'Loading cached data from {self.data_path}.')
            with open(self.data_path, 'r') as f_read:
                self.persona_topic_data = json.load(f_read)

    def _compile_data(self) -> List[List[dict]]:
        """
        Compile data to be saved for faster future use.
        """
        warn_once(f'Starting to compile {self.num_episodes():d} episodes.')
        all_data = []
        for episode_idx in tqdm(range(self.num_episodes())):
            episode_data = []
            entry_idx = 0
            while True:
                example_data = self._get_example(
                    episode_idx=episode_idx, entry_idx=entry_idx
                )
                episode_data.append(example_data)
                if example_data['episode_done']:
                    all_data.append(episode_data)
                    break
                else:
                    entry_idx += 1

        return all_data

    def _get_example(self, episode_idx: int, entry_idx: Optional[int] = None):
        """
        Get example from the base ED teacher and add persona and WoW topic strings.
        """
        gotten = super().get(episode_idx, entry_idx=entry_idx)
        if entry_idx == 0:
            modified_text = self.persona_topicifier.get_modified_text(gotten['text'])
            gotten['text'] = modified_text
        return gotten

    def get(self, episode_idx: int, entry_idx: Optional[int] = None) -> dict:
        """
        Get example from the final data with personas and WoW topic strings.
        """
        if entry_idx is None:
            entry_idx = 0
        return self.persona_topic_data[episode_idx][entry_idx]
