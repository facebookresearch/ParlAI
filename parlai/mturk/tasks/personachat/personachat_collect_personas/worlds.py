#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.core.worlds import validate
from joblib import Parallel, delayed
import numpy as np
import time
import os
import pickle


class PersonaProfileWorld(MTurkOnboardWorld):
    """A world that provides a persona to the MTurkAgent"""
    def __init__(self, opt, mturk_agent):
        self.opt = opt
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_persona_time = opt['max_persona_time']
        self.range_persona = [int(s) for s in opt['range_persona'].split(',')]
        self.n_persona = np.random.randint(
            self.range_persona[0], self.range_persona[1] + 1
        )
        self.episodeDone = False
        self.persona = []
        self.persone_done = False
        super().__init__(opt, mturk_agent)

    def parley(self):
        # Persona creation instructions
        self.mturk_agent.observe({
            'id': 'SYSTEM',
            'text': (
                'Please create your character by entering'
                '<b><span style="color:blue">{} sentences</span></b> below in'
                'the input-box. \nYou have <b><span style="color:blue">{} '
                'mins</span></b> to finish the persona creation.'
                .format(self.n_persona, int(self.max_persona_time / 60))
            )
        })
        while not self.persona_done:
            act = self.mturk_agent.act(timeout=self.max_persona_time)
            # Check timeout
            if act['episode_done']:
                self.episodeDone = True
                return

            candidate_persona = [
                x
                for x in act['text'].split('.'.strip())  # sic.
                if x != ''
            ]

            for cand in candidate_persona:
                # Check if persona is too long
                if len(cand.split(' ')) > 16:
                    control_msg = {
                        'id': 'SYSTEM',
                        'text': (
                            '\n A sentence you entered is too long:\n \
                            <b><span style="color:blue">' + cand +
                            '</span></b>\nPlease resend a sentence '
                            '<b><span style="color:blue">less than 15 '
                            'words</span></b>.'
                        )
                    }
                    self.mturk_agent.observe(validate(control_msg))
                    candidate_persona.remove(cand)
                # Check if persona is too short
                if len(cand.split(' ')) < 3:
                    control_msg = {
                        'id': 'SYSTEM',
                        'text': (
                            '\n A sentence you entered is too short:\n'
                            '<b><span style="color:blue">' + cand +
                            '</span></b>\n Please resend a sentence '
                            '<b><span style="color:blue">more than 3 '
                            'words</span></b>.'
                        )
                    }
                    self.mturk_agent.observe(validate(control_msg))
                    candidate_persona.remove(cand)

            self.persona += candidate_persona

            if len(self.persona) >= self.n_persona:
                self.persona_done = True
                control_msg = {
                    'id': 'SYSTEM',
                    'text': (
                        'Thank you for creating the character! \n '
                        '<b><span style="color:blue">Please click '
                        '"Done with this HIT" below to submit the HIT.'
                        '</span></b>'
                    ),
                    'exceed_min_turns': True
                }
                self.mturk_agent.observe(validate(control_msg))
                self.episodeDone = True

            if not self.persona_done:
                control_msg = {
                    'id': 'SYSTEM',
                    'text': (
                        'Please enter at least *{}* more sentence(s) to finish. '
                        .format(str(self.n_persona - len(self.persona)))
                    )
                }
                self.mturk_agent.observe(validate(control_msg))

    def save_data(self):
        data_path = self.opt['data_path'] + '/personas'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        filename = os.path.join(
            data_path,
            'persona_{}_{}_{}.pkl'.format(
                time.strftime("%Y%m%d-%H%M%S"),
                self.mturk_agent.worker_id,
                self.task_type
            )
        )
        print('Profile successfully saved at {}.'.format(filename))
        pickle.dump({'hit_id': self.mturk_agent.hit_id,
                     'assignment_id': self.mturk_agent.assignment_id,
                     'worker_id': self.mturk_agent.worker_id,
                     'n_persona': self.n_persona,
                     'persona': self.persona}, open(filename, 'wb'))

    def shutdown(self):
        global shutdown_agent

        def shutdown_agent(mturk_agent):
            mturk_agent.shutdown()
        Parallel(
            n_jobs=1,
            backend='threading'
        )(delayed(shutdown_agent)(agent) for agent in [self.mturk_agent])

    def review_work(self):
        if self.persona_done:
            self.mturk_agent.approve_work()

    def episode_done(self):
        return self.episodeDone
