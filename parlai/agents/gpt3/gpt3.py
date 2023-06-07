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
    import openai
except ImportError:
    raise ImportError('Please run `pip install openai`.')


class Gpt3Agent(Agent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        group = parser.add_argument_group('GPT3 Arguments')
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
            help="""Choose a version of GPT-3 that varies on cost, quality, and speed like text-davinci-003, text-davinci-002,
            text-curie-001, text-babbage-001, or text-ada-001""",
        )
        group.add_argument(
            '--init-prompt',
            type=str,
            default='',
            help="""Initial prompt that starts the conversation. Turns of conversation are appended to subsequent OpenAI
            completion queries.""",
        )
        group.add_argument(
            '--suffix',
            type=str,
            help='Suffix that comes after the completion of an inserted text',
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
        self.id = 'Gpt3Agent'
        self.turns = ''
        self.history = FakeHistory(self)
        self.model_name = opt.get('model_name')
        self.init_prompt = opt.get('init_prompt')
        self.suffix = opt.get('suffix')
        self.max_tokens = opt.get('max_tokens')
        self.temperature = opt.get('temperature')
        self.top_p = opt.get('top_p')
        self.stop_sequence = opt.get('stop_sequence')
        self.presence_penalty = opt.get('presence_penalty')
        self.frequency_penalty = opt.get('frequency_penalty')

        openai.api_key = opt.get('openai_api_key')

    def reset(self):
        """
        Reset the agent, clearing its observation.

        Many subclasses implement additional reset logic.
        """
        self.observation = None
        self.turns = ''

    def observe(self, observation):
        """
        Receive an observation/action dict.
        """
        self.observation = observation
        self.turns += f"\n{observation['id']}: {observation['text']}"
        return observation

    def act(self):
        """
        Generate response to last seen observation.
        """
        self.turns += f"\n{self.getID()}: "
        full_prompt = f"{self.init_prompt}{self.turns}"
        resp = self.query_completion_api(full_prompt)
        resp_txt = resp.choices[0].text
        resp_txt = resp_txt.strip()
        self.turns += f"{resp_txt}"
        return Message({'id': self.getID(), 'text': resp_txt, 'episode_done': False})

    def query_completion_api(self, full_prompt):
        response = openai.Completion.create(
            prompt=full_prompt,
            model=self.model_name,
            suffix=self.suffix,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_sequence,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )
        return response


class FakeHistory:
    def __init__(self, gpt3Agent):
        self.gpt3Agent = gpt3Agent

    def add_reply(self, text):
        self.gpt3.turns += f"\n{self.gpt3Agent.getID()}: {text}"

    def get_history_str(self):
        return f"{self.gpt3Agent.init_prompt}{self.gpt3Agent.turns}"
