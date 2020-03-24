#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
import re
from collections import defaultdict
from typing import List, Optional, Dict

from parlai.core.teachers import ParlAIDialogTeacher, create_task_agent_from_taskname
from parlai.tasks.blended_skill_talk.dataset_classification_teacher import (
    AbstractDatasetClassificationTeacher,
)
from parlai.tasks.blended_skill_talk.mixed_candidates_teacher import (
    AbstractMixedCandidatesTeacher,
)
from parlai.tasks.convai2.agents import DefaultTeacher as Convai2DefaultTeacher
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.wizard_of_wikipedia.agents import (
    WizardDialogKnowledgeTeacher,
    BasicdialogTeacher,
)
from parlai.utils.misc import warn_once
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


################################################################################
## Teachers for adding ConvAI2 personas and WoW topics to existing datasets   ##
################################################################################


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


class PersonaTopicifier:
    def __init__(
        self,
        should_have_personas=False,
        should_have_topics=False,
        no_persona_is_error: bool = False,
    ):
        print('IN PERSONA TOPICIFIER INIT')
        self.utterance_to_persona_map = {}
        self.should_have_personas = should_have_personas
        self.should_have_topics = should_have_topics
        self.no_persona_is_error = no_persona_is_error
        # Throw an exception if a persona is not found for the input WoW topic

        # this returns map of persona line str to WoW topic
        (
            self.wow_topics_to_persona_strings_map,
            self.persona_strings_to_wow_topics_map,
        ) = self._setup_personas_to_wow_topics()
        self.personas_file_path = os.path.join(
            opt['datapath'], 'blended_skill_talk', 'persona_list.txt'
        )
        with open(self.personas_file_path, 'r') as f:
            self.personas = f.read().strip().split('||')
            # There's an extra line at the end of the file which is ''
            self.personas = [p for p in self.personas if p]
            print(f'Got {len(self.personas)} personas.')

    def _setup_personas_to_wow_topics(self) -> Dict[str, List[str]]:
        topic_to_persona_path = os.path.join(
            opt['datapath'], 'blended_skill_talk', 'topic_to_persona_list.txt'
        )
        persona_strings_to_topics = defaultdict(list)
        topics_to_persona_strings = defaultdict(list)
        with open(topic_to_persona_path, 'r') as f:
            for line in f:
                match = re.fullmatch(r'([^[]+): (\[.+\])\n', line)
                topic = match.group(1)
                # if topic not in self.wow_topics_to_episode_idxes:
                #     continue
                persona_strings = eval(match.group(2))
                assert isinstance(persona_strings, list)
                topics_to_persona_strings[topic] = persona_strings
                for str_ in persona_strings:
                    persona_strings_to_topics[str_].append(topic)

        print(
            f'FINISHED MAPPING personas to topics, got: {len(list(persona_strings_to_topics.keys()))} persona strings to map to topics.'
        )
        return topics_to_persona_strings, persona_strings_to_topics

    def __calculate_word_overlap(self, a, b):
        """
        Super stupid way of calculating.
        """
        score = 0
        tokens_a = a.split(' ')
        tokens_a = [ta for ta in tokens_a if len(ta) >= 5]
        for ta in tokens_a:
            if ta in b:
                score += 1

        tokens_b = b.split(' ')
        tokens_b = [tb for tb in tokens_b if len(tb) >= 5]
        for tb in tokens_b:
            if tb in a:
                score += 1
        return score

    def __choose_persona_from_text(self, utt):
        utt = utt.strip()
        if utt not in self.utterance_to_persona_map:
            best_word_overlap = 0
            best_persona = None
            for p in self.personas:
                word_overlap = self.__calculate_word_overlap(utt, p)
                if word_overlap >= best_word_overlap:
                    best_word_overlap = word_overlap
                    best_persona = p
            if not best_persona:
                raise Exception(
                    f'No persona found for utterance: \"{utt}\". This should not happen.'
                )
            self.utterance_to_persona_map[utt] = best_persona
            # Should have a \n at the end of it already
            return best_persona
        return self.utterance_to_persona_map[utt]

    def __choose_persona_from_topic(self, topic):
        topic = topic.strip()
        persona_strings = self.wow_topics_to_persona_strings_map[topic]
        for p in persona_strings:
            for persona in self.personas:
                if p in persona:
                    return persona
        if self.no_persona_is_error:
            raise ValueError(f'ERROR: Found no persona for topic: {topic}.')
        else:
            warn_once(
                f'Found no persona for topic: {topic}. Returning first ' f'persona.'
            )
            return self.personas[0]

    def __choose_topic(self, persona):
        persona_lines = persona.strip().split('\n')
        for p in persona_lines:
            p_str = p.replace('your persona:', '')
            p_str = p_str.strip()
            # print(f'Looking for \"{p_str}\" in wow topics map.')
            if p_str in self.persona_strings_to_wow_topics_map:
                topics = self.persona_strings_to_wow_topics_map[p_str]
                topic = topics[0] + '\n'
                # print(f'Found topic: {topic} for persona: \"{persona}\"')
                return topic

        for utt, topics in self.persona_strings_to_wow_topics_map.items():
            utt_words = utt.split()
            utt_words_long = [utt for utt in utt_words if len(utt) > 6]
            for long_utt in utt_words_long:
                if long_utt in persona:
                    # print(f'Returning topic: {topics[0]} for \"{persona}\" after exact match not found.')
                    return topics[0] + '\n'
        warn_once(
            f'Found no WoW topic for persona: \"{persona}\". Returning topics[0]: {topics[0]}'
        )
        return topics[0] + '\n'

    def get_modified_text(self, text):
        # Order should be <Persona> \n <Topic> \n <Utterance>
        # Should be used for entry_idx == 0 only (for all first
        # utterances only)

        # Doesn't work b/c Wizard sometimes sends only the topic
        # has_neither = 'persona:' not in text and '\n' not in text
        # has_wow_topic_only = 'persona:' not in text and '\n' in text
        # has_persona_only = 'persona:' in text

        has_neither = not self.should_have_personas and not self.should_have_topics
        has_wow_topic_only = not self.should_have_personas and self.should_have_topics
        has_persona_only = not self.should_have_topics and self.should_have_personas

        if (self.should_have_personas and (has_neither or has_wow_topic_only)) or (
            self.should_have_topics and (has_neither or has_persona_only)
        ):
            raise Exception(
                f'Malformed text: {text}, should_have_personas: {self.should_have_personas}, should_have_topics: {self.should_have_topics}, has_neither: {has_neither}, has_wow_topic_only: {has_wow_topic_only}, has_persona_only: {has_persona_only}'
            )

        if has_neither:
            # Will occur with ED
            persona = self.__choose_persona_from_text(text)
            topic = self.__choose_topic(persona)
            utt = text
        elif has_wow_topic_only:
            # Will occur with Wizard
            parts = text.strip().split('\n')
            if len(parts) > 1:
                topic = parts[0] + '\n'
                utt = parts[1]
                persona = self.__choose_persona_from_topic(topic)
            else:
                # Only has a topic, no utterance
                topic = parts[0] + '\n'
                utt = ''
                persona = self.__choose_persona_from_topic(topic)
        elif has_persona_only:
            # Will occur with Convai2
            lines = text.strip().split('\n')
            utt = lines[-1]
            persona = ''.join(l + '\n' for l in lines[:-1])
            topic = self.__choose_topic(persona)
        else:
            raise Exception(f'Unknown structure of utterance: {text}')

        modified_utterance = persona + topic + utt
        # print(f'Text was: \"{text}\", now is: \"{modified_utterance}\"')
        return modified_utterance
