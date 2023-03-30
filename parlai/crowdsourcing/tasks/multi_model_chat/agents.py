#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict
import random

from parlai.core.agents import Agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent_from_model_file
from parlai.core.loader import load_agent_module
from parlai.utils.logging import logging

PERSONA_SETTER_AGENT = 'persona-agent'


# The delimiter for the parts of the messages (eg, speaker : timestamp : text)
DEFAULT_SPEAKER_TOKEN_DELIM = ':'
SILENCE_TOKEN = '__SILENCE__'

DECISION_MODEL_OVERRIDES = {
    'interactive_mode': True,
    'skip_generation': False,
    'fp16': True,
    'batchsize': 1,
}

SPEECH_MODEL_OVERRIDES = {
    'interactive_mode': True,
    'skip_generation': False,
    'fp16': True,
    'batchsize': 1,
    'inference': 'beam',
    'beam_size': 3,
    'beam_min_length': 20,
    'beam_block_ngram': 3,
    'beam_context_block_ngram': 3,
}


def flatten_personas(personas: Dict, delim='\n', bb3_format=False):
    personass_str_parts = []
    if bb3_format:
        for i, p in enumerate(personas):
            personass_str_parts.append(f"Person {i+1} is: {p['name']}")
            personass_str_parts.append(f"Person {i+1}'s Persona: {p['persona']}")
    else:
        personass_str_parts.append('__personas__')
        personass_str_parts.extend([f"{p['name']}: {p['persona']}" for p in personas])
        personass_str_parts.append('__end-personas__')

    return delim.join(personass_str_parts)


def flatten_location(location: Dict, delim='\n', bb3_format=False):
    if bb3_format:
        location_str_parts = [
            f'Setting: {location["name"]}',
            f'Description: {location["description"]}',
        ]
    else:
        location_str_parts = [
            '__location__',
            f"{location['name']}: {location['description']}",
            '__end-location__',
        ]

    return delim.join(location_str_parts)


class RandomSpeakerDecicionsAgent(Agent):
    """
    Randomly decides who speaks next.
    """

    def set_characters(self, characters):
        self.characters = characters

    def act(self):
        assert hasattr(self, 'characters'), 'Personas are not set.'
        return {
            'text': random.choice(self.characters),
            'id': 'RandomOrderDecision',
            'episode_done': False,
        }


class MultipartyModelChatAgent(Agent):
    """
    Agent to use in a live chat with human.

    The assumption is that there is only 1 human; all other characters are handled by
    model. We still use the regular observe and act cycles of any other ParlAI agent,
    But after each observe, model decides whose turn is next and if it is the human
    character's turn it responds with silence. Otherwise it uses its utterance
    generation model and generates a text response.
    """

    def __init__(self, opt: Opt, shared=None):
        self.id = 'MultipartyChatAgent'
        self.history = []
        self.utterance_delimiter = opt['utterance_delimiter']
        self.include_speaker_in_context = opt['include_speaker_in_context']
        self.add_speaker_to_context_end = opt['add_speaker_to_context_end']
        self.speaker_token_delimiter = opt['speaker_token_delimiter']
        self.add_personas_to_context = opt['add_personas_to_context']
        self.add_location_to_context = opt['add_location_to_context']
        self.context_format = opt['context_format']

        if not shared:
            self._decision_agent = self._create_decision_agent(opt)
            self._speech_agent = self._create_speech_agent(opt)
        else:
            self._decision_agent = shared['decision_agent'].clone()
            self._speech_agent = shared['speech_agent'].clone()

        super().__init__(opt, shared)

    def share(self):
        """
        Share model parameters.
        """
        shared = super().share()
        shared['decision_agent'] = self._decision_agent
        shared['speech_agent'] = self._speech_agent
        return shared

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group(
            'Multiparty agent for model chat (human evals).'
        )
        agent.add_argument(
            '--decision-agent',
            type=str,
            help='Agent for deciding the next speaker.',
        )
        agent.add_argument(
            '--decision-model-file',
            type=str,
            help='Model file for deciding the next speaker (will be ignored if used with --decision-agent).',
        )
        agent.add_argument(
            '--speech-agent',
            type=str,
            help='Agent for generating the response text.',
        )
        agent.add_argument(
            '--speech-model-file',
            type=str,
            help='Model file for generating the response text (will be ignored if used with --speech-agent).',
        )
        agent.add_argument(
            '--utterance-delimiter',
            type=str,
            default='\n',
            help="A string used to separate each utterance in the context. Defaults to newline. For example, 'A: Hello\nB: Hi there'.",
        )
        agent.add_argument(
            '--include-speaker-in-context',
            type='bool',
            default=True,
            help="Whether to include speaker labels in the context. "
            "For example, message = { text: 'Rachel: Hi' } instead of message = { text: 'Hi' }",
        )
        agent.add_argument(
            '--add-speaker-to-context-end',
            type='bool',
            default=True,
            help='Append the current speaker to the end of each context.',
        )
        agent.add_argument(
            '--speaker-token-delimiter',
            type=str,
            default=DEFAULT_SPEAKER_TOKEN_DELIM,
            help="The token to use to separate the speaker label from the actual utterance in `obs['text']`.",
        )
        agent.add_argument(
            '--add-personas-to-context',
            type=bool,
            default=True,
            help="If true, will add the flattened personas to the contet end.",
        )
        agent.add_argument(
            '--add-location-to-context',
            type=bool,
            default=True,
            help="If true, will add the flattened location to the contet end.",
        )
        agent.add_argument(
            '--context-format',
            type=str,
            default='multilight',
            choices=('bb3', 'multilight', 'light'),
            help="The token to use to separate the speaker label from the actual utterance in `obs['text']`.",
        )
        return parser

    def _create_decision_agent(self, opt):
        logging.info('Creating the decision agent.')
        if opt.get('decision_agent'):
            m = load_agent_module(opt['decision_agent'])
            return m(opt)
        elif 'decision_model_file' in opt:
            return create_agent_from_model_file(
                opt['decision_model_file'], opt_overrides=DECISION_MODEL_OVERRIDES
            )
        else:
            raise ValueError(
                "The opt must have 'decision_agent' or 'decision_model_file'."
            )

    def _create_speech_agent(self, opt):
        logging.info('Creating the speech agent.')
        if opt.get('speech_agent'):
            return load_agent_module(opt['speech_agent'])
        elif 'speech_model_file' in opt:
            return create_agent_from_model_file(
                opt['speech_model_file'], opt_overrides=SPEECH_MODEL_OVERRIDES
            )
        else:
            raise ValueError("The opt must have 'speech_agent' or 'speech_model_file'.")

    def get_context(self, context_format=None):
        """
        Generates the text that goes into each of the models (speaker decision and the
        speech).
        """
        if not context_format:
            context_format = self.context_format

        context_parts = []
        if context_format in ('bb3', 'multilight'):
            use_bb3_format = context_format == 'bb3'
            if self.add_location_to_context:
                context_parts.append(
                    flatten_location(self.location, bb3_format=use_bb3_format)
                )
            if self.add_personas_to_context:
                context_parts.append(
                    flatten_personas(self.personas, bb3_format=use_bb3_format)
                )
        elif context_format == 'light':
            context_parts.append('_task_speech')
            if self.add_location_to_context:
                context_parts.append(f'_setting_name {self.location["name"]}')
                context_parts.append(f'_setting_desc {self.location["description"]}')
            if self.add_personas_to_context:
                context_parts.append(f'_self_name {self.personas[0]["name"]}')
                context_parts.append(f'_self_persona {self.personas[0]["persona"]}')
        else:
            raise ValueError(
                f'The requested context format ("{self.context_format}") is not implemented yet.'
            )

        context_parts.extend(self.history)
        return self.utterance_delimiter.join(context_parts)

    def update_history(self, act):
        utterance_line = act["text"]
        if self.include_speaker_in_context:
            utterance_line = f'{act["id"]}: {utterance_line}'
        self.history.append(utterance_line)

    def get_speaker_index(self, spk):
        assert hasattr(self, 'characters'), 'Personas are not set.'
        return self.characters.index(spk)

    def is_human_turn(self, turn_act):
        # The assumption here is that the character with index 0 is the human.
        spk = turn_act['text'].lower()
        return spk not in self.characters or self.get_speaker_index(spk) == 0

    def is_bot_turn(self, turn_act):
        not self.is_human_turn(turn_act)

    def observe(self, observation):
        if observation['id'] == PERSONA_SETTER_AGENT:
            self.location = observation['location']
            self.personas = observation['personas']
            self.characters = [p['name'].lower() for p in self.personas]
            if hasattr(self._decision_agent, 'set_characters'):
                # The random agent has this.
                self._decision_agent.set_characters(self.characters)
            return

        observation['id'] = self.personas[0]['name']
        self.update_history(observation)

    def get_next_turn(self):
        context = self.get_context(context_format=self.context_format)
        logging.debug(f'The decision model context:{context}')
        self._decision_agent.observe({'text': context, 'episode_done': False})
        next_turn = self._decision_agent.act()
        self._decision_agent.reset()
        return next_turn

    def act(self):
        next_turn = self.get_next_turn()
        speaker = next_turn["text"]
        logging.info(f'The next round assigned to {speaker}')
        if self.is_human_turn(next_turn):
            # Returning empty for passing the turn to the human.
            return {'text': '', 'episode_done': False, 'human_turn': True}
        else:
            context = self.get_context()
            if self.add_speaker_to_context_end:
                context = self.utterance_delimiter.join(
                    [context, f'{speaker}{self.speaker_token_delimiter}']
                )
            logging.debug(f'The speech model context:\n{context}')
            self._speech_agent.observe({'text': context, 'episode_done': False})
            response = self._speech_agent.act()
            self._speech_agent.reset()
            response.force_set('id', speaker)
            self.update_history(response)
            return response
