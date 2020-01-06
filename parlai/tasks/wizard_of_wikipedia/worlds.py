#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import json
import random
import os
import string


from parlai.core.agents import create_agent
from parlai.core.message import Message
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
from parlai.tasks.self_chat.worlds import InteractiveWorld as SelfChatBaseWorld
from parlai.utils.misc import warn_once

from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import (
    KnowledgeRetrieverAgent,
)


NO_TOPIC = '[NO TOPIC]'


class InteractiveWorld(DialogPartnerWorld):
    """
    Interactive world for wizard of wikipedia.

    Used for models trained on the task `-t wizard_of_wikipedia`. Automatically
    retrieves knowledge from Wikipedia based on the conversation history using a TF-IDF
    retriever. Then uses a Transformer-based model to select a checked sentence from
    these retrieved passages.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('WoW Interactive World Args')
        parser.add_argument(
            '--print-checked-sentence',
            type='bool',
            default=True,
            help='Print sentence that the model checks.',
        )

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print('[ loading topics.. ]')
        self.opt = opt
        self._load_topics(opt)
        self.num_topics = opt['num_topics']
        self.cnt = 0
        self.human_agent = self.agents[0]
        self.model_agent = self.agents[1]

        self._set_up_knowledge_agent(opt.get('add_token_knowledge', False))

        self.print_checked_sentence = opt['print_checked_sentence']

    def _set_up_knowledge_agent(self, add_token_knowledge=False):
        from parlai.core.params import ParlaiParser

        parser = ParlaiParser(False, False)
        KnowledgeRetrieverAgent.add_cmdline_args(parser)
        parser.set_params(
            model='projects:wizard_of_wikipedia:knowledge_retriever',
            add_token_knowledge=add_token_knowledge,
        )
        knowledge_opt = parser.parse_args([], print_args=False)
        self.knowledge_agent = create_agent(knowledge_opt)

    def _load_topics(self, opt):
        # Load possible chosen topics
        topics_path = os.path.join(
            opt['datapath'], 'wizard_of_wikipedia', 'topic_splits.json'
        )
        # Get training set topics
        datatype = opt['datatype'].split(':')[0]
        self.topic_list = json.load(open(topics_path, 'rb'))[datatype]

    def _get_new_topic(self):
        random.seed()
        topics = random.sample(self.topic_list, self.num_topics - 1)
        topics.append(NO_TOPIC)
        letters = list(string.ascii_uppercase)[: self.num_topics]
        topic_list = {x: y for x, y in zip(letters, topics)}
        topic_text = '\n'.join(['{}: {}'.format(k, v) for k, v in topic_list.items()])

        done = False
        while not done:
            self.human_agent.observe(
                {
                    'text': '\nPlease choose one of the following topics by typing '
                    'A, B, C, ..., etc. : \n\n{}\n'.format(topic_text)
                }
            )
            topic_act = self.human_agent.act()
            choice = topic_act['text'][0].upper()
            if choice in topic_list:
                done = True
            else:
                self.human_agent.observe(
                    {'text': 'Invalid response, please try again.'}
                )

        chosen_topic = topic_list[choice]
        print('[ Your chosen topic is: {} ]'.format(chosen_topic))
        return chosen_topic

    def _add_knowledge_to_act(self, act):
        self.knowledge_agent.observe(act, actor_id='apprentice')
        knowledge_act = self.knowledge_agent.act()
        act['knowledge'] = knowledge_act['text']
        act['checked_sentence'] = knowledge_act['checked_sentence']
        if self.print_checked_sentence:
            print(
                '[ Using chosen sentence from Wikpedia ]: {}'.format(
                    knowledge_act['checked_sentence']
                )
            )
        act['title'] = knowledge_act['title']
        return act

    def parley(self):
        """
        Loop between wizard and apprentice.

        Adds knowledge to the wizard observations. Assumes that the model agent is the
        wizard model.
        """

        if self.cnt == 0:
            self.topic = self._get_new_topic()
            self.acts = [None, None]
            self.human_first = random.choice([0, 1])

        # possibly get human act first
        if self.cnt == 0 and not self.human_first:
            self.acts[0] = act = Message({'text': '', 'episode_done': False})
            act = self.acts[0]
        else:
            self.acts[0] = self.human_agent.act()
            act = deepcopy(self.acts[0])

        # model agent observe
        if self.cnt == 0 and self.topic != NO_TOPIC:
            # add the chosen_topic to the message
            act['chosen_topic'] = self.topic
            act.force_set('text', '\n'.join([self.topic, act.get('text', 'hi')]))

        # add knowledge to the model observation
        act = self._add_knowledge_to_act(act)

        # model observes knowledge and human (apprentice) act
        self.model_agent.observe(validate(act))

        # model agent act
        self.acts[1] = self.model_agent.act()

        # add the model reply to the knowledge retriever's dialogue history
        self.knowledge_agent.observe(self.acts[1], actor_id='wizard')

        # human (apprentice) agent observes model act
        self.human_agent.observe(validate(self.acts[1]))

        self.update_counters()
        self.cnt += 1

        if self.episode_done():
            print('[ CHAT DONE ]')
            print('\n[ Preparing new chat... ]\n')
            self.cnt = 0
            self.model_agent.reset()


class InteractiveGeneratorWorld(InteractiveWorld):
    """
    Interactive world for generative models.

    Specifically a world for models trained on the task `-t wizard_of_wikipedia
    generator`.
    """

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print('[ loading topics.. ]')
        self.opt = opt
        self._load_topics(opt)
        self.num_topics = opt['num_topics']
        self.cnt = 0
        self.human_agent = self.agents[0]
        self.model_agent = self.agents[1]

        self._set_up_knowledge_agent(add_token_knowledge=True)

    def _add_knowledge_to_act(self, act):
        act = super()._add_knowledge_to_act(act)
        if self.opt.get('prepend_gold_knowledge', False):
            warn_once(
                'Prepending selected knowledge to dialogue input.'
                'If this was not intended behavior, please run with the '
                'flag --prepend-gold-knowledge False'
            )
            knowledge_text = ' '.join(
                [TOKEN_KNOWLEDGE, act['checked_sentence'], TOKEN_END_KNOWLEDGE]
            )
            new_text = '\n'.join([knowledge_text, act['text']])
            act.force_set('text', new_text)
        else:
            warn_once(
                'Not prepending selected knowledge to dialogue input.'
                'If this was not intended behavior, please run with the '
                'flag --prepend-gold-knowledge True'
            )
        return act


class InteractiveSelfchatWorld(SelfChatBaseWorld):
    def init_contexts(self):
        print('[ loading topics.. ]')
        # Load possible chosen topics
        topics_path = os.path.join(
            self.opt['datapath'], 'wizard_of_wikipedia', 'topic_splits.json'
        )
        # Get training set topics
        datatype = self.opt['datatype'].split(':')[0]
        self.topic_list = json.load(open(topics_path, 'rb'))[datatype]

    def get_contexts(self):
        random.seed()
        topic = random.choice(self.topic_list)
        return [topic, topic]
