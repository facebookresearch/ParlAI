#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Knowledge retrieval agent.

Used in interactive mode when the knowledge is not available. Uses the retrieval model
from the model zoo.
"""

from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
from parlai.zoo.wizard_of_wikipedia.knowledge_retriever import download

import json
import os

RETRIEVER_FILE = 'models:wikipedia_full/tfidf_retriever/model'
SELECTOR_FILE = 'models:wizard_of_wikipedia/knowledge_retriever/model'


class KnowledgeRetrieverAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        TorchRankerAgent.add_cmdline_args(argparser)
        parser = argparser.add_argument_group('KnowledgeRetriever Arguments')
        parser.add_argument('--retriever-model-file', type=str, default=RETRIEVER_FILE)
        parser.add_argument('--selector-model-file', type=str, default=SELECTOR_FILE)
        parser.add_argument(
            '--num-retrieved',
            type=int,
            default=7,
            help='how many passages to retrieve for each category',
        )
        parser.add_argument(
            '--add-token-knowledge',
            type='bool',
            default=False,
            help='Add knowledge token to retrieved knowledge',
        )
        parser.add_argument('--debug', type='bool', default=False)
        return parser

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.add_token_knowledge = opt['add_token_knowledge']
        self.model_path = os.path.join(
            opt['datapath'],
            'models',
            'wizard_of_wikipedia',
            'full_dialogue_retrieval_model',
        )

        if not shared:
            # Create retriever
            download(opt['datapath'])  # make sure to download all relevant files
            self._set_up_tfidf_retriever(opt)
            self._set_up_selector(opt)
        else:
            self.selector = create_agent_from_shared(shared['selector'])
            self.retriever = shared['retriever']
            self.sent_tok = shared['sent_tok']
            self.wiki_map = shared['wiki_map']

        self.id = 'KnowledgeRetrieverAgent'

        # NOTE: dialogue history should NOT be shared between instances
        self.retriever_history = {'episode_done': False}
        self.dialogue_history = []
        self.checked_sentence_history = []

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

    def _set_up_selector(self, opt):
        selector_opt = {
            'datapath': opt['datapath'],
            'model_file': opt['selector_model_file'],
            'eval_candidates': 'inline',
            'model': 'transformer/biencoder',
            'batchsize': 1,
            'interactive_mode': True,
            'interactive_candidates': 'inline',
            'override': {'model': 'transformer/biencoder', 'batchsize': 1},
        }
        for k, v in self.opt.items():
            if k not in selector_opt:
                selector_opt[k] = v
        self.selector = create_agent(selector_opt)

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

    def maintain_retriever_history(self, obs, actor_id='apprentice'):
        """
        Maintain texts retrieved by the retriever to mimic the set-up from the data
        collection for the task.
        """

        if actor_id == 'apprentice':
            if self.retriever_history['episode_done']:
                # clear history
                self.retriever_history['chosen_topic'] = ''
                self.retriever_history['wizard'] = ''
                self.retriever_history['apprentice'] = ''
                self.retriever_history['episode_done'] = False
                self.dialogue_history = []

            # save chosen topic
            if 'chosen_topic' in obs:
                self.retriever_history['chosen_topic'] = obs['chosen_topic']
            if 'text' in obs:
                self.retriever_history['apprentice'] = obs['text']

            self.retriever_history['episode_done'] = obs['episode_done']

        elif not self.retriever_history['episode_done']:
            # only add the wizard item to the history if episode_done is not
            # True
            self.retriever_history['wizard'] = obs['text']

        self.dialogue_history.append({'id': actor_id, 'text': obs['text']})

    def get_chosen_topic_passages(self, chosen_topic):
        retrieved_txt_format = []
        retrieved_txt_format_no_title = []
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
                        if self.add_token_knowledge:
                            delim = ' ' + TOKEN_KNOWLEDGE + ' '
                        else:
                            delim = ' '
                        retrieved_txt_format.append(delim.join([chosen_topic, sent]))
                        retrieved_txt_format_no_title.append(sent)
                        total += 1

        if len(retrieved_txt_format) > 0:
            passages = '\n'.join(retrieved_txt_format)
            passages_no_title = '\n'.join(retrieved_txt_format_no_title)
        else:
            passages = ''
            passages_no_title = ''

        return passages, passages_no_title

    def get_passages(self, act):
        """
        Format passages retrieved by taking the first paragraph of the top
        `num_retrieved` passages.
        """
        retrieved_txt = act.get('text', '')
        cands = act.get('text_candidates', [])
        if len(cands) > 0:
            retrieved_txts = cands[: self.opt['num_retrieved']]
        else:
            retrieved_txts = [retrieved_txt]

        retrieved_txt_format = []
        retrieved_txt_format_no_title = []
        for ret_txt in retrieved_txts:
            paragraphs = ret_txt.split('\n')
            if len(paragraphs) > 2:
                sentences = self.sent_tok.tokenize(paragraphs[2])
                for sent in sentences:
                    if self.add_token_knowledge:
                        delim = ' ' + TOKEN_KNOWLEDGE + ' '
                    else:
                        delim = ' '
                    retrieved_txt_format.append(delim.join([paragraphs[0], sent]))
                    retrieved_txt_format_no_title.append(sent)

        if len(retrieved_txt_format) > 0:
            passages = '\n'.join(retrieved_txt_format)
            passages_no_title = '\n'.join(retrieved_txt_format_no_title)
        else:
            passages = ''
            passages_no_title = ''

        return passages, passages_no_title

    def tfidf_retriever_act(self, history):
        """
        Combines and formats texts retrieved by the TFIDF retriever for the chosen
        topic, the last thing the wizard said, and the last thing the apprentice said.
        """
        # retrieve on chosen topic
        chosen_topic_txts = None
        if self.retriever_history.get('chosen_topic'):
            (
                chosen_topic_txts,
                chosen_topic_txts_no_title,
            ) = self.get_chosen_topic_passages(self.retriever_history['chosen_topic'])

        # retrieve on apprentice
        apprentice_txts = None
        if self.retriever_history.get('apprentice'):
            apprentice_act = {
                'text': self.retriever_history['apprentice'],
                'episode_done': True,
            }
            self.retriever.observe(apprentice_act)
            apprentice_txts, apprentice_txts_no_title = self.get_passages(
                self.retriever.act()
            )

        # retrieve on wizard
        wizard_txts = None
        if self.retriever_history.get('wizard'):
            wizard_act = {
                'text': self.retriever_history['wizard'],
                'episode_done': True,
            }
            self.retriever.observe(wizard_act)
            wizard_txts, wizard_txts_no_title = self.get_passages(self.retriever.act())

        # combine everything
        combined_txt = ''
        combined_txt_no_title = ''
        if chosen_topic_txts:
            combined_txt += chosen_topic_txts
            combined_txt_no_title += chosen_topic_txts_no_title
        if wizard_txts:
            combined_txt += '\n' + wizard_txts
            combined_txt_no_title += '\n' + wizard_txts_no_title
        if apprentice_txts:
            combined_txt += '\n' + apprentice_txts
            combined_txt_no_title += '\n' + apprentice_txts_no_title

        return combined_txt, combined_txt_no_title

    def _format_selector_observation(self, knowledge_no_title, episode_done=False):
        obs = {'episode_done': episode_done}
        obs['label_candidates'] = [x for x in knowledge_no_title.split('\n') if x]
        text = self.retriever_history.get('chosen_topic', '')
        if len(self.dialogue_history) > 0:
            if len(self.dialogue_history) > 1:
                text += self.dialogue_history[-2]['text']
            text += self.dialogue_history[-1]['text']
        obs['text'] = text
        return obs

    def _get_checked_sentence(self, knowledge_no_title):
        # choose a sentence from the retrieved knowledge using a
        # Transformer-based ranking model; the downstream dialogue model may
        # or may not make use of this chosen sentence
        selector_obs = self._format_selector_observation(
            knowledge_no_title, episode_done=self.observation.get('episode_done', False)
        )

        self.selector.observe(selector_obs)
        chosen_sentence_act = self.selector.act()
        cands = chosen_sentence_act.get('text_candidates', [])
        if not cands:
            return ''

        for cand in cands:
            if cand not in self.checked_sentence_history:
                self.checked_sentence_history.append(cand)
                return cand

        # return best prediction
        return cands[0]

    def act(self):
        # retrieve knowledge based on the dialogue history using a TF-IDF retriever
        knowledge, knowledge_no_title = self.tfidf_retriever_act(self.retriever_history)

        act = {'text': knowledge, 'episode_done': False, 'id': self.id}
        act['checked_sentence'] = self._get_checked_sentence(knowledge_no_title)
        act['title'] = self.retriever_history.get('chosen_topic', '')

        # add the wizard reply to the dialogue history (this only happens
        # if there are labels or eval_labels -- not in interactive mode)
        if self.next_wizard is not None:
            self.maintain_retriever_history(
                {'text': self.next_wizard}, actor_id='wizard'
            )

        return act

    def observe(self, observation, actor_id='apprentice'):
        """
        Observe a dialogue act.

        Use `actor_id` to indicate whether dialogue acts are from the 'apprentice' or
        the 'wizard' agent.
        """
        if actor_id == 'apprentice':
            self.observation = observation
        self.maintain_retriever_history(observation, actor_id=actor_id)
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
        shared['selector'] = self.selector.share()
        shared['sent_tok'] = self.sent_tok
        shared['wiki_map'] = self.wiki_map
        return shared
