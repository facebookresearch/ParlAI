#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.mturk.core.agents import \
    MTURK_DISCONNECT_MESSAGE, RETURN_MESSAGE, TIMEOUT_MESSAGE

import time


def is_disconnected(act):
    return 'text' in act and \
            act['text'] in [MTURK_DISCONNECT_MESSAGE, RETURN_MESSAGE,
                            TIMEOUT_MESSAGE]


class LightEvalTestWorld(MTurkOnboardWorld):
    '''Task world that gives a pre-determined task as a test. Assigns a
    blocking qualification if the worker fails the test.
    '''

    GESTURES = list(map(lambda x: 'gesture ' + x, [
      'applaud', 'blush', 'cry', 'dance',
      'frown', 'gasp', 'grin', 'groan', 'growl',
      'yawn', 'laugh', 'nod', 'nudge', 'ponder', 'pout', 'scream',
      'shrug', 'sigh', 'smile', 'stare', 'wave', 'wink',
    ]))

    block_act = {
        'id': 'System',
        'text': "FAILED",
        'task_data': {'turn': 'FAILED'},
    }

    def block_loop(self):
        print('Worker {} failed onboarding'.format(self.mturk_agent.worker_id))
        self.mturk_agent.observe(self.block_act)
        self.mturk_agent.mturk_manager.soft_block_worker(
            self.mturk_agent.worker_id)
        act = self.mturk_agent.act()
        while not is_disconnected(act):
            self.mturk_agent.observe(self.block_act)
            act = self.mturk_agent.act()
        return True

    def __init__(self, opt, mturk_agent):
        self.mturk_agent = mturk_agent
        self.opt = opt
        self.did_complete = False
        self.wrong = 0
        self.episodeDone = False

    def parley(self):
        self.mturk_agent.update_agent_id('TestEmote')
        first_act = {
            'id': 'System',
            'text': 'FIRST_TURN',
            'task_data': {
                'wrong': 0,
                'turn': 'FIRST_TURN',
                'actions': self.GESTURES,
                'agent_id': 'Guard',
                'text':
                    'Bahahaha that\'s a great one! Where\'d you get that from?',
                'persona':
                    'I\'m a guard of the royal family. I have a loud laugh, '
                    'and people hear it often as I love jokes. I stand up for '
                    'rightousness, and have a short temper when it comes to '
                    'insults against the king. Sometimes you need to knock '
                    'some sense into people.',
                'base_name': 'Guard',
                'partner_name': 'Jester',
                'setting':
                    'You are in the servants\' quarters. Many people are '
                    'sitting around waiting to be called for services. It\'s '
                    'cozy, but not cramped. A chest is here. A Jester is here. '
                    'You are carrying a spear.',
            }
        }
        self.mturk_agent.observe(first_act)
        act = self.mturk_agent.act()
        if is_disconnected(act):
            self.episodeDone = True
            return
        while act['text'] != 'gesture laugh':
            self.wrong += 1
            if self.wrong > 3:
                return self.block_loop()
            first_act['task_data']['wrong'] = self.wrong
            self.mturk_agent.observe(first_act)
            act = self.mturk_agent.act()
            if is_disconnected(act):
                self.episodeDone = True
                return
        self.mturk_agent.update_agent_id('TestSpeech')
        correct_phrase = (
            'Now you better watch your tongue Jester. '
            'I won\'t have you badmouthing our king.'
        )
        second_act = {
            'id': 'System',
            'text': 'SECOND_TURN',
            'task_data': {
                'wrong': 0,
                'turn': 'SECOND_TURN',
                'curr_message_context': {'action': 'gesture frown'},
                'actions': [
                    'You think you can say whatever you want because we\'re alone?',
                    'Do you want to grab some tea?',
                    'What makes you think you can stand up to me, silly man? I have three times your strength. I have weapons to the teeth. What would make you think this was a good idea?',  # NOQA
                    'Yeah that guy is something of a jerk',
                    'I just feel he doesn\'t have the best sense of humor...',
                    'Yeah landlubber, aye find this is a great hiding spot too.',
                    'If only you could say that to my face one more time. I\'ve missed you too much...',  # NOQA
                    'One more beer for the gang? I feel like you would be the type to have plenty to drink.', # NOQA
                    'The servants quarters are pretty tightly packed aren\'t they?',
                    'I hate being an archer...',
                    correct_phrase,
                    'Once upon a time I lived for that king, but nowadays I feel like I could go without him. Thats why I\'m here in the servants quarters.',  # NOQA
                    'Hey there little fella, do you think you can get me some food?',
                    'I know you want more than just some of our wares, I\'m selling everything.',  # NOQA
                    'One more song! I know you know a few more of them!',
                    'If that isn\'t a good joke, I don\'t know what is? Hahahahaha',
                    'Three fort nights too late, I will not stand for this! You should have been here sooner!',  # NOQA
                    'Aw sweetheart, I just want you to know how much I care.',
                    'I have no spells for you! My wizardry is just for me and my acolytes.',  # NOQA
                    'How did you find out the kinds of jokes that the king likes so much?',  # NOQA
                ]
            }
        }
        self.mturk_agent.observe(second_act)
        act = self.mturk_agent.act()
        if is_disconnected(act):
            self.episodeDone = True
            return

        while act['text'] != correct_phrase:
            self.wrong += 1
            if self.wrong > 3:
                return self.block_loop()
            second_act['task_data']['wrong'] = self.wrong
            self.mturk_agent.observe(second_act)
            act = self.mturk_agent.act()
            if is_disconnected(act):
                self.episodeDone = True
                return
        self.mturk_agent.update_agent_id('TestAct')
        third_act = {
            'id': 'System',
            'text': 'THIRD_TURN',
            'task_data': {
                'wrong': 0,
                'turn': 'THIRD_TURN',
                'text':
                    'You gotta get your senses straight. Hyah! '
                    'Consider this a warning...',
                'actions': [
                    'drop spear',
                    'wield spear',
                    'hug Jester',
                    'examine chest',
                    'get coins from chest',
                    'hit Jester',
                    'steal ball from Jester',
                ]
            }
        }
        self.mturk_agent.observe(third_act)
        act = self.mturk_agent.act()
        if is_disconnected(act):
            self.episodeDone = True
            return
        if act['text'] != 'hit Jester':
            self.wrong += 1
            if self.wrong > 3:
                return self.block_loop()
            third_act['task_data']['wrong'] = self.wrong
            self.mturk_agent.observe(third_act)
            act = self.mturk_agent.act()
            if is_disconnected(act):
                self.episodeDone = True
                return

        self.did_complete = True
        self.mturk_agent.observe({
            'id': 'System',
            'text': 'FINAL_TURN',
            'task_data': {'turn': 'FINAL_TURN', 'wrong': 0}
        })
        self.episodeDone = True
        time.sleep(3)
        return


class LightEvalTaskWorld(MTurkTaskWorld):
    '''Task world steps the worker through a conversation, giving them cands
    to select from as if they are a retrieval model.
    '''

    def __init__(self, opt, mturk_agents, sample, use_train, max_wrong):
        self.mturk_agent = mturk_agents[0]
        self.sample_acts = sample
        self.turn = 0
        self.episodeDone = False
        self.completed = False
        self.selections = []
        self.corrects = [
            ex['labels'][0] if 'labels' in ex else ex['eval_labels']
            for ex in sample
        ]
        self.use_train = use_train
        self.max_wrong = max_wrong

    def extract_from_flag(self, text, flag):
        return text.split(flag)[1]

    def get_current_turn_context(self):
        all_lines = []
        for act in self.sample_acts[:self.turn]:
            lines = act['text'].split('\n')
            if lines[-1].startswith('_self'):
                lines = lines[:-1]
            all_lines += lines

        lines = all_lines + self.sample_acts[self.turn]['text'].split('\n')
        lines = list(filter(lambda x: len(x) > 0, lines))

        setting_name = 'Setting withheld'
        setting_desc = 'Setting description withheld'
        self_name = 'Character withheld'
        partner_name = 'Partner withheld'
        self_persona = 'Persona withheld'
        self_act = ''
        self_text = 'Spoken text withheld'
        messages = []
        self_message = {}
        partner_message = {}

        # Handle current turn context separately
        if lines[-1].startswith('_self'):
            self_line = lines[-1]
            lines = lines[:-1]
            # Extract current turn context
            if self_line.startswith('_self_say'):
                self_text = self.extract_from_flag(self_line, '_self_say')
            elif self_line.startswith('_self_act'):
                self_act = self.extract_from_flag(self_line, '_self_act')
            elif self_line.startswith('_self_emote'):
                self_act = self.extract_from_flag(self_line, '_self_emote')

        # Construct the rest of the context
        for line in lines:
            if line.startswith('_setting_name'):
                setting_name = self.extract_from_flag(line, '_setting_name')
            elif line.startswith('_setting_desc'):
                setting_desc = self.extract_from_flag(line, '_setting_desc')
            elif line.startswith('_partner_name'):
                partner_name = self.extract_from_flag(line, '_partner_name')
            elif line.startswith('_self_name'):
                self_name = self.extract_from_flag(line, '_self_name')
            elif line.startswith('_self_persona'):
                self_persona = self.extract_from_flag(line, '_self_persona')
            elif line.startswith('_partner'):
                if 'id' in self_message:
                    messages.append(self_message)
                    self_message = {}
                if line.startswith('_partner_say'):
                    partner_message['id'] = partner_name
                    partner_message['text'] = \
                        self.extract_from_flag(line, '_partner_say')
                if line.startswith('_partner_act'):
                    partner_message['task_data'] = {
                        'action': self.extract_from_flag(line, '_partner_act')
                    }
                if line.startswith('_partner_emote'):
                    partner_message['task_data'] = {
                        'action': 'gesture ' +
                                  self.extract_from_flag(line, '_partner_emote')
                    }
            elif line.startswith('_self'):
                if 'id' in partner_message:
                    messages.append(partner_message)
                    partner_message = {}
                if line.startswith('_self_say'):
                    self_message['id'] = self_name
                    self_message['text'] = \
                        self.extract_from_flag(line, '_self_say')
                if line.startswith('_self_act'):
                    self_message['task_data'] = {
                        'action': self.extract_from_flag(line, '_self_act')
                    }
                if line.startswith('_self_emote'):
                    self_message['task_data'] = {
                        'action': 'gesture ' +
                                  self.extract_from_flag(line, '_self_emote')
                    }

        if 'id' in partner_message:
            messages.append(partner_message)

        act = {
            'id': 'System',
            'text': 'TASK_DATA',
            'task_data': {
                'actions':
                    sorted(self.sample_acts[self.turn]['label_candidates']),
                'text': self_text,
                'curr_message_context': {'action': self_act},
                'agent_id': self_name,
                'base_name': self_name,
                'persona': self_persona,
                'partner_name': partner_name,
                'setting': setting_desc,
                'setting_name': setting_name,
                'messages': messages
            }
        }
        return act

    def parley(self):
        self.mturk_agent.observe(self.get_current_turn_context())
        act = self.mturk_agent.act()
        if is_disconnected(act):
            self.episodeDone = True
            return
        self.selections.append(act['text'])

        self.turn += 1
        if self.turn == len(self.sample_acts):
            self.episodeDone = True
            self.completed = True
            wrong = 0
            if self.use_train:
                for i in range(len(self.selections)):
                    if self.selections[i] != self.corrects[i]:
                        wrong += 1
            if wrong > self.max_wrong:
                self.completed = False
                self.mturk_agent.mturk_manager.soft_block_worker(
                    self.mturk_agent.worker_id)
                print('Worker failed in train', self.mturk_agent.worker_id)

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.mturk_agent.shutdown()

    def get_custom_task_data(self):
        # brings important data together for the task, to later be used for
        # creating the dataset. If data requires pickling, put it in a field
        # called 'needs-pickle'.
        return {
            'selections': self.selections,
            'corrects': self.corrects,
            'episode': self.sample_acts,
            'training': self.use_train,
        }
