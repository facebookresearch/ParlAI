#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Knowledge retrieval agent. Used in interactive mode when the knowledge is
not available. Uses the retrieval model from the model zoo.
"""

from parlai.core.agents import Agent, create_agent

import json
import os


# TODO: put this guy in the model zoo and replace


class KnowledgeRetrieverAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        parser = argparser.add_argument_group('KnowledgeRetriever Arguments')
        parser.add_argument(
            '--retriever-model-file',
            type=str,
            default='models:wikipedia_full/tfidf_retriever/model',
        )
        parser.add_argument(
            '--num-retrieved',
            type=int,
            default=7,
            help='how many passages to retrieve for each category',
        )
        parser.add_argument(
            '--add-passage-title',
            type='bool',
            default=False,
            help='Add the passage title to the set of retrieved knowledge'
        )
        # TODO: make the above default to True for certain agents
        parser.add_argument('--debug', type='bool', default=False)
        return parser

    def __init__(self, opt, shared=None):
        self.add_passage_title = opt['add_passage_title']

        if not shared:
            # Create retriever
            self._set_up_tfidf_retriever(opt)
        else:
            self.retriever = shared['retriever']
            self.sent_tok = shared['sent_tok']
            self.wiki_map = shared['wiki_map']

        self.id = 'KnowledgeRetrieverAgent'

        # NOTE: dialogue history should NOT be shared between instances
        self.dialogue_history = {}

    def _set_up_tfidf_retriever(self, opt):
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

    def maintain_retrieved_texts(
        self,
        history,
        observation,
        actor_id='apprentice',
    ):
        """
        Maintain texts retrieved by the retriever to mimic the set-up
        from the data collection for the task.
        """

        if actor_id == 'apprentice':
            if history['episode_done']:
                # clear history
                history['chosen_topic'] = ''
                history['wizard'] = ''
                history['apprentice'] = ''
                history['episode_done'] = False

            # save chosen topic
            if 'chosen_topic' in observation:
                history['chosen_topic'] = observation['chosen_topic']
            if 'text' in observation:
                history['apprentice'] = observation['text']

            history['episode_done'] = observation['episode_done']

        elif not history['episode_done']:
            # only add the wizard item to the history if episode_done is not
            # True
            history['wizard'] = observation['text']

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
                        if self.add_passage_title:
                            retrieved_txt_format.append(
                                ' '.join([chosen_topic, sent])
                            )
                        else:
                            retrieved_txt_format.append(sent)
                        total += 1

        if len(retrieved_txt_format) > 0:
            passages = '\n'.join(retrieved_txt_format)
        else:
            passages = ''

        return passages

    def get_passages(self, act):
        """
        Format passages retrieved by taking the first paragraph of the
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
                    if self.add_passage_title:
                        retrieved_txt_format.append(
                            ' '.join([paragraphs[0], sent])
                        )
                    else:
                        retrieved_txt_format.append(sent)

        if len(retrieved_txt_format) > 0:
            passages = '\n'.join(retrieved_txt_format)
        else:
            passages = ''

        return passages

    def tfidf_retriever_act(self, history):
        """
        Combines and formats texts retrieved by the TFIDF retriever for the
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

    def act(self):
        knowledge = self.tfidf_retriever_act(self.dialogue_history)

        # TODO: get the 'golden knowledge' from the knowledge selector agent

        act = {
            'text': knowledge,
            'episode_done': False,
            'id': self.id,
        }
        if self.next_wizard is not None:
            # add the reply to the dialogue history
            self.maintain_retrieved_texts(
                self.dialogue_history,
                {'text': self.next_wizard},
                actor_id='wizard',
            )
        return act

    def observe(self, observation, actor_id='apprentice'):
        """
        Observe a dialogue act. Use `actor_id` to indicate whether dialogue acts
        are from the 'apprentice' or the 'wizard' agent.
        """
        self.maintain_retrieved_texts(
            self.dialogue_history,
            observation,
            actor_id=actor_id
        )
        if 'labels' in observation or 'eval_labels' in observation:
            labels = 'labels' if 'labels' in observation else 'eval_labels'
            self.next_wizard = observation[labels][0]
        else:
            # NOTE: if we are in interactive mode, we need to make sure
            # to call observe again on the wizard's act
            self.next_wizard = None

    def share(self):
        shared = super().share()
        shared['opt'] = self.opt
        shared['retriever'] = self.retriever
        shared['sent_tok'] = self.sent_tok
        shared['wiki_map'] = self.wiki_map
