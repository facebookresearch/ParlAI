#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import json
import random
import os
import string

from parlai.core.message import Message
from parlai.core.worlds import DialogPartnerWorld, validate


NO_TOPIC = '[NO TOPIC]'


class InteractiveWorld(DialogPartnerWorld):
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print('[ loading topics.. ]')
        self.load_topics(opt)
        self.num_topics = opt['num_topics']
        self.cnt = 0
        self.human_agent = self.agents[0]
        self.model_agent = self.agents[1]

    def load_topics(self, opt):
        # Load possible chosen topics
        topics_path = os.path.join(
            opt['datapath'], 'wizard_of_wikipedia', 'topic_splits.json'
        )
        # Get training set topics
        datatype = opt['datatype'].split(':')[0]
        self.topic_list = json.load(open(topics_path, 'rb'))[datatype]

    def get_new_topic(self):
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

    def parley(self):
        if self.cnt == 0:
            self.topic = self.get_new_topic()
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
        self.model_agent.observe(validate(act))

        # model agent act
        self.acts[1] = self.model_agent.act()

        # human agent observe
        self.human_agent.observe(validate(self.acts[1]))

        self.update_counters()
        self.cnt += 1

        if self.episode_done():
            print('[ CHAT DONE ]')
            print('\n[ Preparing new chat... ]\n')
            self.cnt = 0
            self.model_agent.reset()


class InteractiveSelfchatWorld(InteractiveWorld):
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print('[ loading topics.. ]')
        self.load_topics(opt)
        self.num_topics = opt['num_topics']
        self.cnt = 0
        self.model1_agent = self.agents[0]
        self.model2_agent = self.agents[1]

    def get_new_topic(self):
        random.seed()
        return random.choice(self.topic_list)

    def parley(self):
        if self.cnt == 0:
            self.topic = self.get_new_topic()
            self.acts = [None, None]
            # choose speaking order:
            if random.choice([0, 1]):
                self.agents_ordered = [self.agents[0], self.agents[1]]
            else:
                self.agents_ordered = [self.agents[1], self.agents[0]]
                
        acts = self.acts
        agents = self.agents_ordered

        if self.cnt == 0:
            # initial context
            context = {
                'text': self.topic,
                'episode_done': False
            }
            agents[0].observe(validate(context))
            print("TOPIC: " + self.topic)
        # now we do regular loop
        acts[0] = agents[0].act()
        agents[1].observe(validate(acts[0]))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))

        print(self.display() + '\n~~')

        self.update_counters()
        self.cnt += 1

        if self.cnt > 8:
            print('[ CHAT DONE ]')
            print('\n[ Preparing new chat... ]\n')
            self.cnt = 0
            agents[0].reset()
            agents[1].reset()






