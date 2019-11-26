#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent which first retrieves from a database and then reads the dialogue + knowledge from
the database to answer.

NOTE: this model only works for eval, it assumes all training is already done.
"""

from parlai.core.agents import Agent
from parlai.core.agents import create_agent
import regex


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
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('RetrieverReader Arguments')
        agent.add_argument('--retriever-model-file', type=str, default=None)
        agent.add_argument('--reader-model-file', type=str, default=None)
        agent.add_argument(
            '--num-retrieved', type=int, default=5, help='how many passages to retrieve'
        )
        agent.add_argument(
            '--split-paragraphs',
            type='bool',
            default=True,
            help='Whether to split the retrieved passages into ' 'paragraphs',
        )
        return agent

    def observe(self, obs):
        self.retriever.observe(obs)
        self.observation = obs

    def _split_doc(self, doc):
        """
        Given a doc, split it into chunks (by paragraph).
        """
        GROUP_LENGTH = 0
        docs = []
        curr = []
        curr_len = 0
        for split in regex.split(r'\n+', doc):
            split = split.strip()
            if len(split) == 0:
                continue
            # Maybe group paragraphs together until we hit a length limit
            if len(curr) > 0 and curr_len + len(split) > GROUP_LENGTH:
                # yield ' '.join(curr)
                docs.append(' '.join(curr))
                curr = []
                curr_len = 0
            curr.append(split)
            curr_len += len(split)
        if len(curr) > 0:
            # yield ' '.join(curr)
            docs.append(' '.join(curr))
        return docs

    def act(self):
        act_retriever = self.retriever.act()
        obs = self.observation
        obs['episode_done'] = True
        retrieved_txt = act_retriever.get('text', '')
        cands = act_retriever.get('text_candidates', [])
        if len(cands) > 0:
            retrieved_txts = cands[: self.opt['num_retrieved']]
        else:
            retrieved_txts = [retrieved_txt]
        text = obs['text']
        reader_acts = []
        retrieved_txts = [r for r in retrieved_txts if r != '']
        for ret_txt in retrieved_txts:
            if self.opt.get('split_paragraphs', False):
                paragraphs = self._split_doc(ret_txt)
            else:
                paragraphs = [ret_txt]
            for para in paragraphs:
                obs['text'] = para + '\n' + text
                self.reader.observe(obs)
                act_reader = self.reader.act()
                act_reader['paragraph'] = para
                reader_acts.append(act_reader)
        if len(reader_acts) > 0:
            best_act = max(reader_acts, key=lambda x: x['candidate_scores'][0])
        else:
            best_act = {'id': self.getID()}

        return best_act
