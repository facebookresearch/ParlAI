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
from projects.wizard_of_wikipedia.wizard_transformer_ranker.wizard_transformer_ranker import (
    WizardTransformerRankerAgent,
)

import json
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
            self.retriever = shared['retriever']
            self.responder = create_agent_from_shared(shared['responder_shared_opt'])
            self.sent_tok = shared['sent_tok']
            self.wiki_map = shared['wiki_map']

        self.id = 'WizardRetrievalInteractiveAgent'
        self.ret_history = {}

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        WizardTransformerRankerAgent.add_cmdline_args(argparser)
        parser = argparser.add_argument_group('WizardRetrievalInteractive Arguments')
        parser.add_argument(
            '--retriever-model-file',
            type=str,
            default='models:wikipedia_full/tfidf_retriever/model',
        )
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
        parser.add_argument(
            '--num-retrieved',
            type=int,
            default=7,
            help='how many passages to retrieve for each' 'category',
        )
        parser.add_argument('--debug', type='bool', default=False)
        return parser

    def _set_up_retriever(self, opt):
        retriever_opt = {
            'model_file': opt['retriever_model_file'],
            'remove_title': False,
            'datapath': opt['datapath'],
            'override': {'remove_title': False},
        }
        self.retriever = create_agent(retriever_opt)

        self._set_up_sent_tok()
        wiki_map_path = os.path.join(self.model_path, 'chosen_topic_to_passage.json')
        self.wiki_map = json.load(open(wiki_map_path, 'r'))

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

    def _set_up_sent_tok(self):
        try:
            import nltk
        except ImportError:
            raise ImportError('Please install nltk (e.g. pip install nltk).')
        # nltk-specific setup
        st_path = 'tokenizers/punkt/{0}.pickle'.format('english')
        try:
            self.sent_tok = nltk.data.load(st_path)
        except LookupError:
            nltk.download('punkt')
            self.sent_tok = nltk.data.load(st_path)

    def get_chosen_topic_passages(self, chosen_topic):
        retrieved_txt_format = []
        if chosen_topic in self.wiki_map:
            retrieved_txt = self.wiki_map[chosen_topic]
            retrieved_txts = retrieved_txt.split('\n')

            if len(retrieved_txts) > 1:
                combined = ' '.join(retrieved_txts[2:])
                sentences = self.sent_tok.tokenize(combined)
                total = 0
                for sent in sentences:
                    if total >= 10:
                        break
                    if len(sent) > 0:
                        retrieved_txt_format.append(' '.join([chosen_topic, sent]))
                        total += 1

        if len(retrieved_txt_format) > 0:
            passages = '\n'.join(retrieved_txt_format)
        else:
            passages = ''

        return passages

    def get_passages(self, act):
        """Format passages retrieved by taking the first paragraph of the
        top `num_retrieved` passages.
        """
        retrieved_txt = act.get('text', '')
        cands = act.get('text_candidates', [])
        if len(cands) > 0:
            retrieved_txts = cands[: self.opt['num_retrieved']]
        else:
            retrieved_txts = [retrieved_txt]

        retrieved_txt_format = []
        for ret_txt in retrieved_txts:
            paragraphs = ret_txt.split('\n')
            if len(paragraphs) > 2:
                sentences = self.sent_tok.tokenize(paragraphs[2])
                for sent in sentences:
                    retrieved_txt_format.append(' '.join([paragraphs[0], sent]))

        if len(retrieved_txt_format) > 0:
            passages = '\n'.join(retrieved_txt_format)
        else:
            passages = ''

        return passages

    def retriever_act(self, history):
        """Combines and formats texts retrieved by the TFIDF retriever for the
        chosen topic, the last thing the wizard said, and the last thing the
        apprentice said.
        """
        # retrieve on chosen topic
        chosen_topic_txts = None
        if self.ret_history.get('chosen_topic'):
            chosen_topic_txts = self.get_chosen_topic_passages(
                self.ret_history['chosen_topic']
            )

        # retrieve on apprentice
        apprentice_txts = None
        if self.ret_history.get('apprentice'):
            apprentice_act = {
                'text': self.ret_history['apprentice'],
                'episode_done': True,
            }
            self.retriever.observe(apprentice_act)
            apprentice_txts = self.get_passages(self.retriever.act())

        # retrieve on wizard
        wizard_txts = None
        if self.ret_history.get('wizard'):
            wizard_act = {'text': self.ret_history['wizard'], 'episode_done': True}
            self.retriever.observe(wizard_act)
            wizard_txts = self.get_passages(self.retriever.act())

        # combine everything
        combined_txt = ''
        if chosen_topic_txts:
            combined_txt += chosen_topic_txts
        if wizard_txts:
            combined_txt += '\n' + wizard_txts
        if apprentice_txts:
            combined_txt += '\n' + apprentice_txts

        return combined_txt

    def observe(self, observation):
        obs = observation.copy()
        self.maintain_retrieved_texts(self.ret_history, obs)
        if self.debug:
            print('DEBUG: Retriever history:\n{}'.format(self.ret_history))
        responder_knowledge = self.retriever_act(self.ret_history)
        obs['knowledge'] = responder_knowledge
        self.observation = obs

    def maintain_retrieved_texts(self, history, observation):
        """Maintain texts retrieved by the retriever to mimic the set-up
        from the data collection for the task.
        """
        if 'chosen_topic' not in history:
            history['episode_done'] = False
            history['chosen_topic'] = ''
            history['wizard'] = ''
            history['apprentice'] = ''

        if history['episode_done']:
            history['chosen_topic'] = ''
            history['wizard'] = ''
            history['apprentice'] = ''
            if 'next_wizard' in history:
                del history['next_wizard']
            history['episode_done'] = False

        # save chosen topic
        if 'chosen_topic' in observation:
            history['chosen_topic'] = observation['chosen_topic']
        if 'text' in observation:
            history['apprentice'] = observation['text']
        if 'next_wizard' in history:
            history['wizard'] = history['next_wizard']
        # save last thing wizard said (for next time)
        if 'labels' in observation:
            history['next_wizard'] = observation['labels'][0]
        elif 'eval_labels' in observation:
            history['next_wizard'] = observation['eval_labels'][0]

        history['episode_done'] = observation['episode_done']

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
        return responder_act

    def share(self):
        """Share internal saved_model between parent and child instances."""
        shared = super().share()
        shared['opt'] = self.opt
        shared['retriever'] = self.retriever
        shared['responder_shared_opt'] = self.responder.share()
        shared['sent_tok'] = self.sent_tok
        shared['wiki_map'] = self.wiki_map

        return shared
