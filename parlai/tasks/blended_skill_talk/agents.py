#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
from typing import List, Optional

from parlai.core.teachers import ParlAIDialogTeacher, create_task_agent_from_taskname
from parlai.tasks.convai2.agents import DefaultTeacher as Convai2DefaultTeacher
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.wizard_of_wikipedia.agents import (
    WizardDialogKnowledgeTeacher,
    BasicdialogTeacher,
)
from parlai_internal.projects.blended_skill_talk.add_personas_topics import (
    PersonaTopicifier,
)
from parlai.tasks.blended_skill_talk.dataset_classification_teacher import (
    AbstractDatasetClassificationTeacher,
)
from parlai_internal.tasks.blended_skill_talk.mixed_candidates_teacher import (
    AbstractMixedCandidatesTeacher,
)
from .build import build


##################################################
#### Teacher for the BlendedSkillTalk Dataset ####
##################################################


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'blended_skill_talk', dt + '.txt')


class BlendedSkillTalkTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _path(opt)
        super().__init__(opt, shared)


################################################################################
## Teachers for BlendedSkillTalk Two-Stage Model                              ##
## 1) Input for the classifier to decide which dataset an utterance is from   ##
## 2) Mixed Candidates evaluation                                             ##
################################################################################


class BSTBuilder(object):
    @staticmethod
    def create_three_dataset_teachers_map(opt):
        # Instantiate the Wizard teacher which will set up the data for you
        # Puts dataset at <datapath>/wizard_of_wikipedia/' if needed to build
        # The wizard teacher annoying changes the task so need to copy the opt
        wizard_opt = copy.deepcopy(opt)
        wizard_opt['task'] = 'wizard_of_wikipedia:random_split'
        wizard_teacher = WizardDialogKnowledgeTeacher(wizard_opt)

        # PersonaChat has self original (only one side personas),  or both
        # original (see both personas). Default is only self.
        personachat_teacher = Convai2DefaultTeacher(opt)

        # Puts the dataset at <datapath>/empatheticdialogues
        # We should train the model on both sides of the conversation
        ed_opt = copy.deepcopy(opt)
        ed_opt['train_experiencer_only'] = False
        empatheticdialogue_teacher = EmpatheticDialoguesTeacher(ed_opt)

        print('[BSTBuilder] Returning num examples(): ')
        teachers_map = {
            '__WIZARD__': wizard_teacher,
            '__PERSONA__': personachat_teacher,
            '__EMPATHETIC__': empatheticdialogue_teacher,
        }
        for dataset_label, teacher in teachers_map.items():
            print(f'{dataset_label}: {teacher.num_examples()}')
        return teachers_map

    @staticmethod
    def get_fixed_candidates_path(opt, dataset_label):
        return f'{opt["datapath"]}/blended_skill_talk/{dataset_label}_utterances_{opt["datatype"]}.cands'

    @staticmethod
    def build_fixed_candidates(opt):
        """
        Gather all the labels in the current datetype and write to a file if not
        already there (otherwise skip)
        This is then used in a two stage model to instantiate pretrained instances
        """
        teachers_map = BSTBuilder.create_three_dataset_teachers_map(opt)
        for dset_label, dset_task_teacher in teachers_map.items():
            fixed_cands_file = BSTBuilder.get_fixed_candidates_path(opt, dset_label)
            BSTBuilder._build_fixed_candidates(dset_task_teacher, fixed_cands_file)

    @staticmethod
    def _build_fixed_candidates(task_teacher, fixed_candidates_file):
        """
        Gathering all the labels in the current datetype and write to a file
        Which is then used in the model
        """
        if os.path.isfile(fixed_candidates_file):
            print(f'Candidates file: {fixed_candidates_file} exists. Returning')
            return
        candidates = []
        _episode_idx = 0

        while _episode_idx < task_teacher.num_episodes():
            _entry_idx = 0
            while True:
                ex = task_teacher.get(_episode_idx, _entry_idx)
                candidates.append(ex['labels'][0])
                if ex['episode_done']:
                    break
                _entry_idx += 1
            _episode_idx += 1

        os.makedirs(os.path.dirname(fixed_candidates_file), exist_ok=True)
        with open(fixed_candidates_file, "w+") as f:
            # in labels I do not think there are any newlines
            # unlike in the actual utterances
            s = ''.join([c + '\n' for c in candidates])
            f.write(s)
        print(
            f'Wrote {len(candidates)} to {fixed_candidates_file} for teacher: {task_teacher}'
        )


class BST3DatasetClassificationTeacher(AbstractDatasetClassificationTeacher):
    """
    Subclasses AbstractDatasetClassificationTeacher which separates examples
    from a list of input teachers into their own episodes and sets the label to
    be name of the dataset
    """

    def create_dataset_teachers_map(self, opt):
        """
        Implement the superclass' abstract method
        :return: map of {<dataset label>: <teacher>}
        """
        return BSTBuilder.create_three_dataset_teachers_map(opt)

    def preprocess_utterance(self, utterance):
        """
        Implement superclass' abstract method
        Do something to utterance before including
        We remove lines with \n (topics for Wizard or personas for persona chat)
        """
        return utterance.split('\n')[-1].strip()

    def build(self, opt):
        # This is needed for model evaluation when pretrained polyencoder models
        # instantiated with the fixed candidate set
        BSTBuilder.build_fixed_candidates(opt)
        super().build(opt)


class BSTMixedCandidatesTeacher(AbstractMixedCandidatesTeacher):
    def create_dataset_teachers_map(self, opt):
        """
        Implement the superclass' abstract method
        :return: map of {<dataset label>: <teacher>}
        """
        return BSTBuilder.create_three_dataset_teachers_map(opt)

    def get_mixed_candidates_path(self, opt):
        """Where to look for or store the mixed candidates calculated"""
        return os.path.join(
            opt['datapath'],
            'blended_skill_talk',
            f'mixed_candidates_{self.__class__.__name__}_{self.mc_task}_{self.opt["datatype"]}.cands',
        )


class InteractiveTeacher(BlendedSkillTalkTeacher):
    # Dummy class to add arguments for interactive world.
    pass


class SelfchatTeacher(BlendedSkillTalkTeacher):
    # Dummy class to add arguments for interactive world.
    pass


class DefaultTeacher(BlendedSkillTalkTeacher):
    pass


def create_agents(opt):
    if not opt.get('interactive_task', False):
        return create_task_agent_from_taskname(opt)
    else:
        # interactive task has no task agents (they are attached as user agents)
        return []


class ConvAI2PersonaTopicifierTeacher(Convai2DefaultTeacher):
    """
    Adds WoW topics to ConvAI2 data.
    """

    def __init__(self, opt, shared=None):
        self.persona_topicifier = PersonaTopicifier(
            should_have_personas=True, should_have_topics=False
        )
        super().__init__(opt, shared=shared)

    def get(self, episode_idx, entry_idx=None):
        gotten = super().get(episode_idx, entry_idx=entry_idx)
        if entry_idx == 0:
            modified_text = self.persona_topicifier.get_modified_text(gotten['text'])
            gotten['text'] = modified_text
        return gotten


class WoWPersonaTopicifierTeacher(WizardDialogKnowledgeTeacher):
    """
    Adds personas to WoW data.
    """

    def __init__(self, opt, shared=None):
        self.persona_topicifier = PersonaTopicifier(
            should_have_personas=False, should_have_topics=True
        )
        super().__init__(opt, shared=shared)

    def get(self, episode_idx, entry_idx=None):
        gotten = super().get(episode_idx, entry_idx=entry_idx)
        if entry_idx == 0:
            modified_text = self.persona_topicifier.get_modified_text(gotten['text'])
            gotten['text'] = modified_text
        return gotten


class WoWBasicPersonaTopicifierTeacher(BasicdialogTeacher):
    """
    Adds personas to WoW data.
    """

    def __init__(self, opt, shared=None):
        if opt.get('add_topic', False) is not True:
            raise ValueError(
                'add_topic must be True so that BasicPersonaTopicifierTeacher only '
                'needs to add personas.'
            )
        self.persona_topicifier = PersonaTopicifier(
            should_have_personas=False, should_have_topics=True
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

    def __init__(self, opt, shared=None):
        self.persona_topicifier = PersonaTopicifier(
            should_have_personas=False, should_have_topics=False
        )
        super().__init__(opt, shared=shared)

        if (
            self.opt.get('deepmoji') is not None
            or self.opt.get('fasttextloc') is not None
            or self.opt.get('prepend', -1) > 0
        ):
            raise NotImplementedError(
                'Using deepmoji or fasttextloc not supported with this teacher.'
            )

        # Running over all examples is really slow because the process of finding a WoW
        # topic is expensive, so let's cache data with personas and topics
        side_string = 'experiencer_only' if self.experiencer_side_only else 'both_sides'
        self.cached_data_path = os.path.join(
            self.opt['datapath'],
            'empatheticdialogues',
            'persona_topicifier',
            f'{self.datatype}__{side_string}.json',
        )
        os.makedirs(os.path.dirname(self.cached_data_path), exist_ok=True)
        if not os.path.isfile(self.cached_data_path):
            print(f'Cached data file at {self.cached_data_path} not found! Creating...')
            self.persona_topic_data = self._compile_data()
            print(f'Saving data to {self.cached_data_path}.')
            with open(self.cached_data_path, 'w') as f_write:
                json.dump(self.persona_topic_data, f_write)
        else:
            print(f'Loading cached data from {self.cached_data_path}.')
            with open(self.cached_data_path, 'r') as f_read:
                self.persona_topic_data = json.load(f_read)

    def _compile_data(self) -> List[List[dict]]:
        """
        Compile data to be saved for faster future use.
        """
        print(f'Starting to compile {self.num_episodes():d} episodes.')
        all_data = []
        for episode_idx in range(self.num_episodes()):
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
            if (episode_idx + 1) % 100 == 0:
                print(f'Compiled {episode_idx+1:d} episodes.')
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
