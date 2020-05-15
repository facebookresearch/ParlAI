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
from tqdm import tqdm

from parlai.core.opt import Opt
from parlai.core.teachers import (
    ParlAIDialogTeacher,
    create_task_agent_from_taskname,
    MultiTaskTeacher,
)
from parlai.tasks.convai2.agents import DefaultTeacher as Convai2DefaultTeacher
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.wizard_of_wikipedia.agents import WizardDialogKnowledgeTeacher
from parlai.utils.misc import warn_once
from .build import build


##################################################
#### Teacher for the BlendedSkillTalk Dataset ####
##################################################


def raw_data_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'blended_skill_talk', dt + '.json')


def _processed_data_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'blended_skill_talk', dt + '.txt')


def _cached_data_path(opt: Opt, experiencer_side_only: bool) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    side_string = 'experiencer_only' if experiencer_side_only else 'both_sides'
    return os.path.join(
        opt['datapath'],
        'blended_skill_talk',
        f'ed_persona_topicifier__{dt}__{side_string}.json',
    )


def safe_personas_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    return os.path.join(opt['datapath'], 'blended_skill_talk', 'safe_personas.txt')


class BlendedSkillTalkTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _processed_data_path(opt)
        super().__init__(opt, shared)


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
            gotten['text'] = modified_text
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


class PersonaTopicifier:
    def __init__(
        self,
        opt: Opt,
        should_have_personas: bool = False,
        should_have_topics: bool = False,
        no_persona_is_error: bool = False,
    ):
        self.datapath = opt['datapath']
        self.utterance_to_persona_map = {}
        self.should_have_personas = should_have_personas
        self.should_have_topics = should_have_topics
        self.no_persona_is_error = no_persona_is_error
        # Throw an exception if a persona is not found for the input WoW topic

        # Get persona files
        build(opt)

        # this returns map of persona line str to WoW topic
        (
            self.wow_topics_to_persona_strings_map,
            self.persona_strings_to_wow_topics_map,
        ) = self._setup_personas_to_wow_topics()
        self.personas_file_path = os.path.join(
            self.datapath, 'blended_skill_talk', 'persona_list.txt'
        )
        with open(self.personas_file_path, 'r') as f:
            self.personas = f.read().strip().split('||')
            # There's an extra line at the end of the file which is ''
            self.personas = [p for p in self.personas if p]

    def _setup_personas_to_wow_topics(self) -> Dict[str, List[str]]:
        topic_to_persona_path = os.path.join(
            self.datapath, 'blended_skill_talk', 'topic_to_persona_list.txt'
        )
        persona_strings_to_topics = defaultdict(list)
        topics_to_persona_strings = defaultdict(list)
        with open(topic_to_persona_path, 'r') as f:
            for line in f:
                match = re.fullmatch(r'([^[]+): (\[.+\])\n', line)
                topic = match.group(1)
                persona_strings = eval(match.group(2))
                assert isinstance(persona_strings, list)
                topics_to_persona_strings[topic] = persona_strings
                for str_ in persona_strings:
                    persona_strings_to_topics[str_].append(topic)

        warn_once(
            f'FINISHED MAPPING personas to topics, got: {len(list(persona_strings_to_topics.keys()))} persona strings to map to topics.'
        )
        return topics_to_persona_strings, persona_strings_to_topics

    def __calculate_word_overlap(self, a, b):
        """
        Very rudimentary way to calculate word overlap.
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
            warn_once(f'Found no persona for topic: {topic}. Returning first persona.')
            return self.personas[0]

    def __choose_topic(self, persona):
        persona_lines = persona.strip().split('\n')
        for p in persona_lines:
            p_str = p.replace('your persona:', '')
            p_str = p_str.strip()
            if p_str in self.persona_strings_to_wow_topics_map:
                topics = self.persona_strings_to_wow_topics_map[p_str]
                topic = topics[0] + '\n'
                return topic

        for utt, topics in self.persona_strings_to_wow_topics_map.items():
            utt_words = utt.split()
            utt_words_long = [utt for utt in utt_words if len(utt) > 6]
            for long_utt in utt_words_long:
                if long_utt in persona:
                    return topics[0] + '\n'

        return topics[0] + '\n'

    def get_modified_text(self, text):
        # Order should be <Persona> \n <Topic> \n <Utterance>
        # Should be used for entry_idx == 0 only (for all first
        # utterances only)

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
        return modified_utterance


class AllTeacher(MultiTaskTeacher):
    """
    Multitask teacher that combines all "Persona Topicifier" teachers.
    """

    def __init__(self, opt, shared=None):
        topicifier_tasks = [
            'blended_skill_talk:ConvAI2PersonaTopicifier',  # ConvAI2
            'blended_skill_talk:EDPersonaTopicifier',  # Empathetic Dialogues
            'blended_skill_talk:WoWPersonaTopicifier',  # Wizard of Wikipedia
            'blended_skill_talk:BlendedSkillTalk',  # Blended Skill Talk
        ]
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(topicifier_tasks)
        super().__init__(opt, shared)
