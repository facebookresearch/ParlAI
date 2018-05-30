# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Agent which first retrieves from a database and then reads the dialogue + knowledge
from the database to answer.
NOTE: this model only works for eval, it assumes all training is already done.
"""

from parlai.core.agents import Agent
from parlai.core.agents import create_agent


class RetrieverReaderAgent(Agent):

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'RetrieverReaderAgent'

        # Create retriever
        retriever_opt = {'model_file': opt['retriever_model_file']}
        self.retriever = create_agent(retriever_opt)

        # Create reader
        reader_opt = {'model_file': opt['reader_model_file']}
        self.reader = create_agent(reader_opt)

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('RetrieverReader Arguments')
        agent.add_argument('--retriever-model-file', type=str, default=None)
        agent.add_argument('--reader-model-file', type=str, default=None)
        return agent

    def observe(self, obs):
        self.retriever.observe(obs)
        self.observation = obs

    def act(self):
        act_retriever = self.retriever.act()
        obs = self.observation
        retrieved_txt = act_retriever.get('text', '')
        if retrieved_txt != '':
            obs['text'] = retrieved_txt + '\n' + obs['text']
        self.reader.observe(obs)
        act_reader = self.retriever.act()
        return act_reader
