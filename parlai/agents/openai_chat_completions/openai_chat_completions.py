#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.message import Message

try:
    import litellm
except ImportError:
    raise ImportError('Please run `pip install litellm`.')


class OpenaiChatCompletionsAgent(Agent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        group = parser.add_argument_group('Chat Completion Arguments')
        group.add_argument(
            '--openai-api-key',
            type=str,
            required=True,
            help='Add your OpenAI api key',
        )
        group.add_argument(
            '--model-name',
            type=str,
            required=True,
            help="""Choose model name like GPT-4 or GPT 3.5""",
        )
        group.add_argument(
            '--init-prompt',
            type=str,
            default='',
            help="""Initial prompt that starts the conversation. Turns of conversation are appended to subsequent OpenAI
            completion queries.""",
        )
        group.add_argument(
            '--role',
            type=str,
            default='assistant',
            choices=['user', 'system', 'assistant'],
            help='Role of the author of message',
        )
        group.add_argument(
            '--counterparty-role',
            type=str,
            default='assistant',
            choices=['user', 'system', 'assistant'],
            help='Role of the other speaker',
        )
        group.add_argument(
            '--name',
            type=str,
            help='Name of the author of the message. Alphanumeric (with underscores) strings up to 64 chars are allowed',
        )
        group.add_argument(
            '--counterparty-name',
            type=str,
            help='Name of the other speaker. Alphanumeric (with underscores) strings up to 64 chars are allowed',
        )
        group.add_argument(
            '--max-tokens',
            type=int,
            required=True,
            help='The max number of tokens generated as a single conversation turn',
        )
        group.add_argument(
            '--temperature',
            type=float,
            default=1.0,
            help="""Temperature ranges between 0-2 such that higher temperature will make outputs more random while lower
            values make the output more deterministic""",
        )
        group.add_argument(
            '--top-p',
            type=float,
            default=1.0,
            help='Determines nucleus sampling rate',
        )
        group.add_argument(
            '--stop-sequence',
            type=str,
            help='Stop sequence is a string that will stop further generation of tokens',
        )
        group.add_argument(
            '--presence-penalty',
            type=float,
            default=0.0,
            help="""Presence penalty ranges between -2.0 to 2.0 such that more positive values will reduce chance of generating
            tokens that already appeared in text""",
        )
        group.add_argument(
            '--frequency-penalty',
            type=float,
            default=0.0,
            help="""Frequency penalty ranges between -2.0 to 2.0 such that more positive values will reduce chance of
            generating tokens that appear frequently""",
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'OpenaiChatCompletionsAgent'
        self.turns = []
        self.history = FakeHistory(self)
        self.model_name = opt.get('model_name')
        self.init_prompt = opt.get('init_prompt')
        self.role = opt.get('role')
        self.counterparty_role = opt.get('counterparty_role')
        self.name = opt.get('name')
        self.counterparty_name = opt.get('counterparty_name')
        self.max_tokens = opt.get('max_tokens')
        self.temperature = opt.get('temperature')
        self.top_p = opt.get('top_p')
        self.stop_sequence = opt.get('stop_sequence')
        self.presence_penalty = opt.get('presence_penalty')
        self.frequency_penalty = opt.get('frequency_penalty')

        # check that string self.init_prompt is not empty nor None
        if self.init_prompt:
            self.turns.append({'role': 'system', 'content': self.init_prompt})

        litellm.api_key = opt.get('openai_api_key')

    def reset(self):
        """
        Reset the agent, clearing its observation.

        Many subclasses implement additional reset logic.
        """
        self.observation = None
        self.turns = []

    def observe(self, observation):
        """
        Receive an observation/action dict.
        """
        self.observation = observation

        if self.observation['id'] == 'context':
            msg = {'role': 'system', 'content': observation['text']}
        else:
            msg = {'role': self.counterparty_role, 'content': observation['text']}
            if self.counterparty_name:
                msg['name'] = self.counterparty_name
        self.turns.append(msg)

        return observation

    def act(self):
        """
        Generate response to last seen observation.
        """
        resp = self.query_chat_completion_api()
        resp_txt = resp['choices'][0]['message']['content']

        msg = {'role': self.role, 'content': resp_txt}
        if self.name:
            msg['name'] = self.name
        self.turns.append(msg)

        return Message({'id': self.getID(), 'text': resp_txt, 'episode_done': False})

    def query_chat_completion_api(self):
        response = litellm.completion(
            model=self.model_name,
            messages=self.turns,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_sequence,
            max_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )
        return response


class FakeHistory:
    def __init__(self, agent):
        self.agent = agent

    def add_reply(self, text, role='assistant', name=None):
        msg = {'role': role, 'content': text}
        if name:
            msg['name'] = name
        self.agent.turns.append()

    def get_history_str(self):
        return str(self.agent.turns)
