# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.core.worlds import validate, MultiAgentDialogWorld
from joblib import Parallel, delayed
import numpy as np
import time
import os
import pickle
import random


class PersonaProfileWorld(MTurkOnboardWorld):
    '''A world that provides a persona to the MTurkAgent'''
    def __init__(self, opt, mturk_agent):
        self.opt = opt
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_persona_time = opt['max_persona_time']
        self.range_persona = [int(s) for s in opt['range_persona'].split(',')]
        self.n_persona = np.random.randint(self.range_persona[0], self.range_persona[1]+1)
        self.episodeDone = False
        mturk_agent.n_persona = self.n_persona
        mturk_agent.persona = []
        super().__init__(opt, mturk_agent)

    def parley(self):
        self.mturk_agent.observe({
            'id': 'SYSTEM',
            'text': 'Please create your persona by entering at least {} sentences below in the input-box. \
                     You have <b>{} mins</b> to finish the persona creation.'.format(self.n_persona, int(self.max_persona_time/60))})
        persona_done = False
        while not persona_done:
            act = self.mturk_agent.act(timeout=self.max_persona_time)
            # timeout
            if act['episode_done'] == True:
                self.episodeDone = True
                return

            self.mturk_agent.persona += list(filter(lambda x: x !='', act['text'].split('.')))

            if len(self.mturk_agent.persona) >= self.n_persona:
                persona_done = True
                control_msg = {'id': 'SYSTEM',
                               'text': 'Thank you for creating the persona! Please wait while we match you with another worker...'}
                self.mturk_agent.observe(validate(control_msg))
                self.episodeDone = True
            if not persona_done:
                control_msg = {'id': 'SYSTEM',
                               'text': 'Please enter at least *{}* more sentence(s) to finish. '.format(str(self.n_persona-len(self.mturk_agent.persona)))}
                self.mturk_agent.observe(validate(control_msg))

    def save_data(self):
        data_path = self.opt['data_path'] + '/personas'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        filename = os.path.join(data_path, 'persona_{}_{}_{}.pkl'.format(time.strftime("%Y%m%d-%H%M%S"), self.mturk_agent.worker_id, self.task_type))
        print('Profile successfully saved at {}.'.format(filename))
        pickle.dump({'hit_id': self.mturk_agent.hit_id,
                     'assignment_id': self.mturk_agent.assignment_id,
                     'worker_id': self.mturk_agent.worker_id,
                     'n_persona': self.n_persona,
                     'persona': self.mturk_agent.persona}, open(filename, 'wb'))

    def episode_done(self):
        return self.episodeDone


class PersonaChatWorld(MultiAgentDialogWorld):
    def __init__(self, opt, agents=None, shared=None,
                 range_turn=[2], max_turn=10,
                 max_resp_time=120,
                 world_tag='NONE',
                 agent_timeout_shutdown=120):
        self.opt = opt
        self.agents=agents
        self.turn_idx = 0
        self.range_turn = range_turn
        self.max_turn = max_turn
        self.n_turn = np.random.randint(self.range_turn[0], self.range_turn[1]) + 1
        self.dialog = []
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.chat_done = False
        self.world_tag = world_tag

        # below are timeout protocols
        self.max_resp_time = max_resp_time # in secs
        self.agent_timeout_shutdown = agent_timeout_shutdown
        super().__init__(opt, agents, shared)

    def parley(self):
        self.turn_idx += 1

        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))

        '''If at first turn, we need to give each agent the instructions'''
        if self.turn_idx == 1:
            for idx, agent in enumerate(self.agents):
                control_msg['text'] = self.get_instruction(idx, tag='start')
                agent.observe(validate(control_msg))

        '''If we get to the min turns, inform turker that they can end if they
           want
        '''
        if self.turn_idx == self.n_turn + 1:
            for idx, agent in enumerate(self.agents):
                control_msg['text'] = self.get_instruction(idx, tag='exceed_min_turns')
                control_msg['exceed_min_turns'] = True
                agent.observe(validate(control_msg))

        '''Otherwise, we proceed accordingly'''
        acts = [None, None]
        for idx, agent in enumerate(self.agents):
            if not self.chat_done:
                acts[idx] = agent.act(timeout=self.max_resp_time)
            if self.check_timeout(acts[idx]):
                return

            if acts[idx]['episode_done'] == True:
                self.chat_done = True
                for ag in self.agents:
                    # if agent disconnected
                    if ag != agent and ag.some_agent_disconnected:
                        control_msg['text'] = 'The other worker unexpectedly diconnected. \
                            Please click "Done with this HIT" button below to finish this HIT.'
                        control_msg['episode_done'] = True
                        ag.observe(validate(control_msg))
                        return
                # agent ends chat after exceeding minimum number of turns
                if self.turn_idx > self.n_turn:
                    for ag in self.agents:
                        ag.observe(validate(acts[idx]))
                        control_msg['text'] = 'One of you ended the chat. Thanks for your time! Please click "Done with this HIT" button below to finish this HIT.'
                        control_msg['episode_done'] = True
                        ag.observe(validate(control_msg))
                return

            else:
                self.dialog.append((idx, acts[idx]['text']))
                for other_agent in self.agents:
                    if other_agent != agent:
                        other_agent.observe(validate(acts[idx]))


    def episode_done(self):
        return self.chat_done

    def get_instruction(self, agent_idx=None, agent_id=None, tag='first'):
        if tag == 'start':
            return '\nSuccessfully matched. Now let\'s get to know each other through the chat! \n\
                    You need to finish at least <b>' + str(self.n_turn) + ' chat turns</b>, \
                    after that you can click the "Done" button to end the chat. \
                    You have ' + str(int(self.max_resp_time/60.)) + ' min(s) for sending each message. \
		    \n <b>Please try to speak to the other person as if you are the persona you created.</b> \n'

        if tag == 'timeout':
            return '{} is timeout. \
                    Please click the "Done with this HIT" button below to finish this HIT.'.format(agent_id)

        if tag == 'exceed_min_turns':
            return '\n {} chat turns finished! \n \
                    Now if it\'s your turn, you can click the "Done" button to end the chat, \
                    or you can keep chatting for more bonus.'.format(self.n_turn, self.max_turn)

    def save_data(self):
        data_path = self.opt['data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        filename = os.path.join(data_path, '{}_{}_{}.pkl'.format(time.strftime("%Y%m%d-%H%M%S"), np.random.randint(0, 1000), self.task_type))
        print(self.world_tag+': Data successfully saved at {}.'.format(filename))
        pickle.dump({'personas': [ag.persona for ag in self.agents],
                     'dialog': self.dialog,
                     'n_turn': self.n_turn,
                     'n_personas': [ag.n_persona for ag in self.agents]}, open(filename, 'wb'))

    def reset_random(self):
        self.n_turn = np.random.randint(self.range_turn[0], self.range_turn[1]) + 1

    def check_timeout(self, act):
        if act['text'] == '[TIMEOUT]' and act['episode_done'] == True:
            control_msg = {'episode_done': True}
            control_msg['id'] = 'SYSTEM'
            control_msg['text'] = self.get_instruction(agent_id=act['id'], tag='timeout')
            for ag in self.agents:
                if ag.id != act['id']:
                    ag.observe(validate(control_msg))
            self.chat_done = True
            return True
        else:
            return False

    def shutdown(self):
        global shutdown_agent
        def shutdown_agent(mturk_agent):
            mturk_agent.shutdown()
        Parallel(
            n_jobs=len(self.agents),
            backend='threading'
        )(delayed(shutdown_agent)(agent) for agent in self.agents)

    def review_work(self):
        global review_agent
        def review_agent(ag):
            bns = 0.
            if len(ag.persona) >= ag.n_persona:
                bns_turn = (self.turn_idx - 1)*0.03 if self.turn_idx < self.max_turn else (self.max_turn)*0.03
                bns += bns_turn
                bns = float('{0:.2f}'.format(bns))

                if bns > 1e-6:
                    ag.pay_bonus(bns)
                ag.approve_work()
        Parallel(n_jobs=len(self.agents), backend='threading')(delayed(review_agent)(agent) for agent in self.agents)
