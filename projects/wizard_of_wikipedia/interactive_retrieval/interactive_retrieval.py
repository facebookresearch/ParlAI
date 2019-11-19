#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Wizard agent with 2 parts:
1. TFIDF retriever (optional, task may already provide knowledge)
2. Retrieval model, retrieves on possible responses and conditions on
   retrieved knowledge

NOTE: this model only works for eval, it assumes all training is already done.
"""

from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import (
    KnowledgeRetrieverAgent,
)
from projects.wizard_of_wikipedia.wizard_transformer_ranker.wizard_transformer_ranker import (
    WizardTransformerRankerAgent,
)

import os


class InteractiveRetrievalAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.debug = opt['debug']
        self.get_unique = opt['get_unique']
        if self.get_unique:
            self.used_messages = []
        self.model_path = os.path.join(
            opt['datapath'],
            'models',
            'wizard_of_wikipedia',
            'full_dialogue_retrieval_model',
        )

        if not shared:
            # Create responder
            self._set_up_responder(opt)
            # Create retriever
            self._set_up_retriever(opt)
        else:
            self.opt = shared['opt']
            self.responder = create_agent_from_shared(shared['responder_shared_opt'])
            self.retriever = create_agent_from_shared(shared['retriever_shared_opt'])

        self.id = 'WizardRetrievalInteractiveAgent'
        self.ret_history = {}

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        WizardTransformerRankerAgent.add_cmdline_args(argparser)
        KnowledgeRetrieverAgent.add_cmdline_args(argparser)
        parser = argparser.add_argument_group('WizardRetrievalInteractive Arguments')
        parser.add_argument(
            '--responder-model-file',
            type=str,
            default='models:wizard_of_wikipedia/full_dialogue_retrieval_model/model',
        )
        parser.add_argument(
            '--get-unique',
            type='bool',
            default=True,
            help='get unique responses from the bot',
        )
        parser.add_argument('--debug', type='bool', default=False)
        return parser

    def _set_up_retriever(self, opt):
        self.retriever = KnowledgeRetrieverAgent(opt)

    def _set_up_responder(self, opt):
        responder_opts = opt.copy()
        # override these opts to build the responder model
        override_opts = {
            'model_file': opt['responder_model_file'],
            'datapath': opt['datapath'],
            'model': 'projects:wizard_of_wikipedia:wizard_transformer_ranker',
            'fixed_candidates_path': os.path.join(self.model_path, 'wizard_cands.txt'),
            'eval_candidates': 'fixed',
            'n_heads': 6,
            'ffn_size': 1200,
            'embeddings_scale': False,
            'delimiter': ' __SOC__ ',
            'n_positions': 1000,
            'legacy': True,
            'no_cuda': True,
            'encode_candidate_vecs': True,
            'batchsize': 1,
            'interactive_mode': True,
        }
        for k, v in override_opts.items():
            responder_opts[k] = v
            responder_opts['override'][k] = v
        self.responder = create_agent(responder_opts)

    def observe(self, observation):
        obs = observation.copy()
        self.retriever.observe(obs, actor_id='apprentice')
        knowledge_act = self.retriever.act()
        if self.debug:
            print('DEBUG: Retrieved knowledge: {}'.format(knowledge_act['texts']))
        obs['knowledge'] = knowledge_act['text']
        self.observation = obs

    def get_unique_reply(self, act):
        # iterate through text candidates until we find a reply that we
        # have not used yet
        for txt in act['text_candidates']:
            if txt not in self.used_messages:
                self.used_messages.append(txt)
                return txt

    def act(self):
        obs = self.observation
        # choose a knowledge sentence
        responder_obs = obs.copy()
        if self.debug:
            print('DEBUG: Responder is observing:\n{}'.format(responder_obs))
        self.responder.observe(responder_obs)
        responder_act = self.responder.act()
        if self.debug:
            print('DEBUG: Responder is acting:\n{}'.format(responder_act))
        responder_act.force_set('id', 'WizardRetrievalInteractiveAgent')
        if self.get_unique:
            responder_act.force_set('text', self.get_unique_reply(responder_act))

        # update the retriever agent history with a self act if necessary
        if 'labels' not in obs and 'eval_labels' not in obs:
            self.retriever.observe(responder_act, actor_id='wizard')

        return responder_act

    def share(self):
        """Share internal saved_model between parent and child instances."""
        shared = super().share()
        shared['opt'] = self.opt
        shared['responder_shared_opt'] = self.responder.share()
        shared['retriever_shared_opt'] = self.retriever.share()
        return shared
