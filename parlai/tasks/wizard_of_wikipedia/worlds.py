#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import json
import random
import os

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.zoo.wizard_of_wikipedia.full_dialogue_retrieval_model import download


class InteractiveWorld(DialogPartnerWorld):
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print("[ loading topics.. ]")
        download(opt['datapath'])
        self.load_topics(opt)
        self.cnt = 0
        self.acts = [None, None]

    def load_topics(self, opt):
        # Load possible chosen topics
        topics_path = os.path.join(
            opt['datapath'],
            'wizard_of_wikipedia',
            'topic_splits.json',
        )
        # Get training set topics
        self.topic_list = json.load(open(topics_path, 'rb'))['train']

    def get_new_topic(self):
        random.seed()
        topics = random.sample(self.topic_list, 3)
        topic_list = {
            'A': topics[0],
            'B': topics[1],
            'C': topics[2],
        }

        topic_text = '\n'.join(['{}: {}'.format(k, v) for k, v in topic_list.items()])

        done = False
        while not done:
            self.agents[0].observe({
                'text': '\nPlease choose one of the following topics by typing '
                        'A, B, or C: \n\n{}\n'.format(topic_text)
            })
            topic_act = self.agents[0].act()
            choice = topic_act['text'][0].upper()
            if choice in topic_list:
                done = True
            else:
                self.agents[0].observe({
                    'text': 'Invalid response, please try again.'
                })

        chosen_topic = topic_list[choice]
        print('[ Your chosen topic is: {} ]'.format(chosen_topic))
        return chosen_topic

    def parley(self):
        """Agent 0 goes first. Alternate between the two agents."""
        if self.cnt == 0:
            self.topic = self.get_new_topic()
        self.acts[0] = self.agents[0].act()
        act = deepcopy(self.acts[0])
        if self.cnt == 0:
            # add the chosen_topic to the message
            act['chosen_topic'] = self.topic
            act['text'] = '\n'.join([self.topic, act.get('text', 'hi')])
        self.agents[1].observe(validate(act))
        self.acts[1] = self.agents[1].act()
        self.agents[0].observe(validate(self.acts[1]))
        self.update_counters()
        self.cnt += 1

        if self.episode_done():
            print('CHAT DONE')
            print('\n...preparing new chat...\n')
            self.cnt = 0
