#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import random
import re
from collections import defaultdict
from typing import List, Optional, Dict, Tuple

from parlai.core.opt import Opt
from parlai.core.teachers import ParlAIDialogTeacher, create_task_agent_from_taskname
from parlai.tasks.convai2.agents import BothTeacher
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.wizard_of_wikipedia.agents import WizardDialogKnowledgeTeacher
from parlai.utils.misc import warn_once
from parlai.utils.io import PathManager
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


def _persona_list_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    return os.path.join(opt['datapath'], 'blended_skill_talk', 'persona_list.txt')


def _topic_to_persona_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    return os.path.join(
        opt['datapath'], 'blended_skill_talk', 'topic_to_persona_list.txt'
    )


def _cached_data_path(opt: Opt, experiencer_side_only: bool) -> str:
    """
    Build the data if it doesn't exist.

    See EDPersonaTopicifierTeacher in ParlAI v1.5.1 and earlier for the code to add
    persona strings to the base EmpatheticDialogues dataset.
    """
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


class PersonaTopicifier:
    def __init__(
        self,
        opt: Opt,
        should_have_personas: bool = False,
        should_have_topics: bool = False,
        no_persona_is_error: bool = False,
    ):
        self.utterance_to_persona_map = {}
        self.should_have_personas = should_have_personas
        self.should_have_topics = should_have_topics
        self.no_persona_is_error = no_persona_is_error
        # Throw an exception if a persona is not found for the input WoW topic

        # this returns map of persona line str to WoW topic
        self.personas_file_path = _persona_list_path(opt)
        self.topic_to_persona_path = _topic_to_persona_path(opt)
        (
            self.wow_topics_to_persona_strings_map,
            self.persona_strings_to_wow_topics_map,
        ) = self._setup_personas_to_wow_topics()
        with PathManager.open(self.personas_file_path, 'r') as f:
            self.personas = f.read().strip().split('||')
            # There's an extra line at the end of the file which is ''
            self.personas = [p for p in self.personas if p]

    def _setup_personas_to_wow_topics(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        persona_strings_to_topics = defaultdict(list)
        topics_to_persona_strings = defaultdict(list)
        with PathManager.open(self.topic_to_persona_path, 'r') as f:
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


################################################################
##  Generator of context for crowdsourcing BST conversations  ##
################################################################


class ContextGenerator:
    """
    Generates contexts shown to crowdsourced workers when collecting BST conversations.

    This generator was used to generate the context information shown to workers at the
    beginning of a conversation, when crowdsourcing the conversations that make up the
    BST dataset.
    """

    def __init__(self, opt, datatype: str = 'train', seed: Optional[int] = None):
        """
        Initialize the context generator.

        opt: only a 'datapath' key is required, to specify the ParlAI data folder
        """

        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

        convai2_opt = Opt({'datapath': opt['datapath'], 'datatype': datatype})
        self.convai2_teacher = BothTeacher(convai2_opt)

        ed_opt = Opt(
            {
                'datapath': opt['datapath'],
                'datatype': datatype,
                'train_experiencer_only': True,
            }
        )
        # Specify train_experiencer_only = True because we want to ensure that the text
        # will correspond to a Speaker utterance and the label to a Listener response
        self.ed_teacher = EmpatheticDialoguesTeacher(ed_opt)

        wow_opt = Opt({'datapath': opt['datapath'], 'datatype': datatype})
        self.wow_teacher = WizardDialogKnowledgeTeacher(wow_opt)

        self.topic_to_persona_path = _topic_to_persona_path(opt)
        self.wow_topics_to_episode_idxes = self._setup_topics_to_episodes()
        self.persona_strings_to_wow_topics = self._setup_personas_to_topics()

    def get_context(self) -> dict:
        """
        Get context information to be shown at the beginning of one conversation.

        Values in return dict:
        - context_dataset: the dataset (ConvAI2, EmpatheticDialogues, or Wizard of
            Wikipedia) used to generate the context information.
        - persona_1_strings, persona_2_strings: 2 persona strings each for the two
            speakers, chosen randomly from the ConvAI2 dataset. If context_dataset ==
            "wizard_of_wikipedia", these persona strings will be matched to the WoW
            topic returned in the "additional_context" field.
        - additional_context: provides additional bits of information to give context
            for the speakers. If context_dataset == "empathetic_dialogues", this is a
            situation from the start of an ED conversation. If context_dataset ==
            "wizard_of_wikipedia", this is a topic from the WoW dataset that matches the
            persona strings. If context_dataset == "convai2", this is None.
        - person1_seed_utterance, person2_seed_utterance: two lines of a conversation
            from the dataset specified by "context_dataset". They will be shown to the
            speakers to "seed" the conversation, and the speakers continue from where
            the lines left off.
        """

        # Determine which dataset we will show context for
        rand_value = self.rng.random()
        if rand_value < 1 / 3:
            context_dataset = 'convai2'
        elif rand_value < 2 / 3:
            context_dataset = 'empathetic_dialogues'
        else:
            context_dataset = 'wizard_of_wikipedia'

        if context_dataset == 'convai2':

            # Select episode
            episode_idx = self.rng.randrange(self.convai2_teacher.num_episodes())

            # Extract personas
            persona_1_strings, persona_2_strings = self._extract_personas(episode_idx)

            # Sample persona strings
            selected_persona_1_strings = self.rng.sample(persona_1_strings, 2)
            selected_persona_2_strings = self.rng.sample(persona_2_strings, 2)

            # Select previous utterances
            num_entries = len(self.convai2_teacher.data.data[episode_idx])
            entry_idx = self.rng.randrange(1, num_entries)
            # Don't select the first entry, which often doesn't include an apprentice
            # utterance
            chosen_entry = self.convai2_teacher.get(episode_idx, entry_idx=entry_idx)
            person1_seed_utterance = chosen_entry['text']
            assert len(chosen_entry['labels']) == 1
            person2_seed_utterance = chosen_entry['labels'][0]

            return {
                'context_dataset': context_dataset,
                'persona_1_strings': selected_persona_1_strings,
                'persona_2_strings': selected_persona_2_strings,
                'additional_context': None,
                'person1_seed_utterance': person1_seed_utterance,
                'person2_seed_utterance': person2_seed_utterance,
            }

        elif context_dataset == 'empathetic_dialogues':

            # Select episode
            persona_episode_idx = self.rng.randrange(
                self.convai2_teacher.num_episodes()
            )

            # Extract personas
            persona_1_strings, persona_2_strings = self._extract_personas(
                persona_episode_idx
            )

            # Sample persona strings
            selected_persona_1_strings = self.rng.sample(persona_1_strings, 2)
            selected_persona_2_strings = self.rng.sample(persona_2_strings, 2)

            # Select previous utterances
            episode_idx = self.rng.randrange(self.ed_teacher.num_episodes())
            entry_idx = 0  # We'll only use the first pair of utterances
            entry = self.ed_teacher.get(episode_idx, entry_idx=entry_idx)
            situation = entry['situation']
            speaker_utterance = entry['text']
            assert len(entry['labels']) == 1
            listener_response = entry['labels'][0]

            return {
                'context_dataset': context_dataset,
                'persona_1_strings': selected_persona_1_strings,
                'persona_2_strings': selected_persona_2_strings,
                'additional_context': situation,
                'person1_seed_utterance': speaker_utterance,
                'person2_seed_utterance': listener_response,
            }

        elif context_dataset == 'wizard_of_wikipedia':

            # Pull different personas until you get a pair for which at least one
            # sentence has a WoW topic bound to it
            num_tries = 0
            while True:

                num_tries += 1

                # Extract a random (matched) pair of personas
                persona_episode_idx = self.rng.randrange(
                    self.convai2_teacher.num_episodes()
                )
                all_persona_strings = dict()
                all_persona_strings[1], all_persona_strings[2] = self._extract_personas(
                    persona_episode_idx
                )

                # See if any of the persona strings have a matching WoW topic
                matching_persona_string_idxes = []
                for persona_idx, persona_strings in all_persona_strings.items():
                    for str_idx, str_ in enumerate(persona_strings):
                        wow_topics = self.persona_strings_to_wow_topics[str_]
                        if len(wow_topics) > 0:
                            matching_persona_string_idxes.append((persona_idx, str_idx))
                if len(matching_persona_string_idxes) > 0:
                    break

            print(
                f'{num_tries:d} try/tries needed to find a pair of personas with an '
                f'associated WoW topic.'
            )

            # Pick out the WoW topic and matching persona string
            matching_persona_idx, matching_persona_string_idx = self.rng.sample(
                matching_persona_string_idxes, k=1
            )[0]
            matching_persona_string = all_persona_strings[matching_persona_idx][
                matching_persona_string_idx
            ]
            wow_topic = self.rng.sample(
                self.persona_strings_to_wow_topics[matching_persona_string], k=1
            )[0]

            # Sample persona strings, making sure that we keep the one connected to the
            # WoW topic
            if matching_persona_idx == 1:
                remaining_persona_1_strings = [
                    str_
                    for str_ in all_persona_strings[1]
                    if str_ != matching_persona_string
                ]
                selected_persona_1_strings = [
                    matching_persona_string,
                    self.rng.sample(remaining_persona_1_strings, k=1)[0],
                ]
                self.rng.shuffle(selected_persona_1_strings)
                selected_persona_2_strings = self.rng.sample(all_persona_strings[2], 2)
            else:
                selected_persona_1_strings = self.rng.sample(all_persona_strings[1], 2)
                remaining_persona_2_strings = [
                    str_
                    for str_ in all_persona_strings[2]
                    if str_ != matching_persona_string
                ]
                selected_persona_2_strings = [
                    matching_persona_string,
                    self.rng.sample(remaining_persona_2_strings, k=1)[0],
                ]
                self.rng.shuffle(selected_persona_2_strings)

            # Sample WoW previous utterances, given the topic
            episode_idx = self.rng.sample(
                self.wow_topics_to_episode_idxes[wow_topic], k=1
            )[0]
            entry_idx = 1
            # Select the second entry, which (unlike the first entry) will always have
            # two valid utterances and which will not usually be so far along in the
            # conversation that the new Turkers will be confused
            entry = self.wow_teacher.get(episode_idx, entry_idx=entry_idx)
            apprentice_utterance = entry['text']
            assert len(entry['labels']) == 1
            wizard_utterance = entry['labels'][0]

            return {
                'context_dataset': context_dataset,
                'persona_1_strings': selected_persona_1_strings,
                'persona_2_strings': selected_persona_2_strings,
                'additional_context': wow_topic,
                'person1_seed_utterance': apprentice_utterance,
                'person2_seed_utterance': wizard_utterance,
            }

    def _setup_personas_to_topics(self) -> Dict[str, List[str]]:
        """
        Create a map from ConvAI2 personas to WoW topics that they correspond to.
        """

        print('Starting to map personas to topics.')

        persona_strings_to_topics = defaultdict(list)
        with PathManager.open(self.topic_to_persona_path, 'r') as f:
            for line in f:
                match = re.fullmatch(r'([^[]+): (\[.+\])\n', line)
                topic = match.group(1)
                if topic not in self.wow_topics_to_episode_idxes:
                    continue
                persona_strings = eval(match.group(2))
                assert isinstance(persona_strings, list)
                for str_ in persona_strings:
                    persona_strings_to_topics[str_].append(topic)

        print('Finished mapping personas to topics.')

        return persona_strings_to_topics

    def _setup_topics_to_episodes(self) -> Dict[str, List[int]]:
        """
        Create a map from WoW topics to the indices of the WoW episodes that use them.
        """

        print('Starting to map topics to episodes.')

        topics_to_episodes = defaultdict(list)
        for episode_idx in range(self.wow_teacher.num_episodes()):
            topic = self.wow_teacher.get(episode_idx, entry_idx=0)['chosen_topic']
            topics_to_episodes[topic].append(episode_idx)

        print('Finished mapping topics to episodes.')

        return topics_to_episodes

    def _extract_personas(self, episode_idx: str) -> Tuple[List[str], List[str]]:
        """
        For the given ConvAI2 conversation, return strings of both speakers' personas.
        """
        first_entry = self.convai2_teacher.get(episode_idx, entry_idx=0)
        first_text_strings = first_entry['text'].split('\n')
        persona_1_strings = []
        persona_2_strings = []
        for str_ in first_text_strings[:-1]:  # The last string is the first utterance
            if str_.startswith('your persona: '):  # Here, "you" are Person 2
                persona_2_strings.append(str_[len('your persona: ') :])
            elif str_.startswith("partner's persona: "):
                persona_1_strings.append(str_[len("partner's persona: ") :])
            else:
                raise ValueError('Persona string cannot be parsed!')
        return persona_1_strings, persona_2_strings
