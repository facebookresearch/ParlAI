#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from joblib import Parallel, delayed
from extract_and_save_personas import main as main_extract
import numpy as np
import time
import os
import pickle
import random

ONBOARD_MSG = '\nWelcome! Below is your persona \
        (you can find it on the left side of the chat)\n \
        When you are ready to start your conversation, \
        click the "I am ready, continue" button below\n'
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
WAITING_MSG = 'Please wait while we match you with another worker...'


class PersonasGenerator(object):
    def __init__(self, opt):
        self.personas_idx_stack_path = os.path.join(
            opt['extract_personas_path'], './personas_idx_stack.pkl'
        )

        self.personas_path = '{}/personas-{}'.format(
            opt['extract_personas_path'],
            opt['persona_type'] + 'Revised' if opt['revised'] else 'Original',
        )

        if not os.path.exists(self.personas_path):
            opt['personas_path'] = self.personas_path
            main_extract(opt)
        self.personas_name_list = []

        for f_name in os.listdir(self.personas_path):
            if f_name.endswith('.pkl'):
                self.personas_name_list.append(f_name)

        if os.path.exists(self.personas_idx_stack_path):
            with open(self.personas_idx_stack_path, 'rb') as handle:
                self.idx_stack = pickle.load(handle)
        else:
            self.idx_stack = []
            self.add_idx_stack()
            self.save_idx_stack()
        pass

    def add_idx_stack(self):
        stack = [i for i in range(len(self.personas_name_list))]
        random.seed()
        random.shuffle(stack)
        self.idx_stack = stack + self.idx_stack

    def pop_persona(self):
        if len(self.idx_stack) == 0:
            self.add_idx_stack()
        idx = self.idx_stack.pop()
        data = np.load(
            os.path.join(self.personas_path, self.personas_name_list[int(idx)]),
            allow_pickle=True,
        )
        return (idx, data)

    def push_persona(self, idx):
        self.idx_stack.append(idx)

    def save_idx_stack(self):
        with open(self.personas_idx_stack_path, 'wb') as handle:
            pickle.dump(self.idx_stack, handle)


class PersonaProfileWorld(MTurkOnboardWorld):
    """
    A world that provides a persona to the MTurkAgent.
    """

    def __init__(self, opt, mturk_agent):
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_persona_time = opt['max_persona_time']
        super().__init__(opt, mturk_agent)

    def parley(self):
        persona_idx, data = self.mturk_agent.persona_generator.pop_persona()
        self.mturk_agent.persona_idx = persona_idx
        self.mturk_agent.persona_data = data
        persona_text = ''
        for s in data:
            persona_text += '<b><span style="color:blue">' '{}\n</span></b>'.format(
                s.strip()
            )

        self.mturk_agent.observe(
            {
                'id': 'SYSTEM',
                'show_persona': True,
                'text': ONBOARD_MSG + '<br>' + persona_text + '<br>',
            }
        )

        act = self.mturk_agent.act(timeout=self.max_persona_time)

        # timeout
        if act['episode_done'] or (('text' in act and act['text'] == TIMEOUT_MESSAGE)):

            self.mturk_agent.persona_generator.push_persona(
                self.mturk_agent.persona_idx
            )
            self.mturk_agent.persona_generator.save_idx_stack()
            self.episodeDone = True
            return

        if 'text' not in act:
            control_msg = {'id': 'SYSTEM', 'text': WAITING_MSG}
            self.mturk_agent.observe(validate(control_msg))
            self.episodeDone = True


class PersonaChatWorld(MultiAgentDialogWorld):
    def __init__(
        self,
        opt,
        agents=None,
        shared=None,
        range_turn=(4, 7),
        max_turn=10,
        max_resp_time=120,
        world_tag='NONE',
        agent_timeout_shutdown=120,
    ):
        self.agents = agents
        self.turn_idx = 0
        self.range_turn = range_turn
        self.max_turn = max_turn
        self.n_turn = np.random.randint(self.range_turn[0], self.range_turn[1]) + 1
        self.dialog = []
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.chat_done = False
        self.world_tag = world_tag

        # below are timeout protocols
        self.max_resp_time = max_resp_time  # in secs
        self.agent_timeout_shutdown = agent_timeout_shutdown
        super().__init__(opt, agents, shared)

        # get personas
        self.personas = [
            (ag.persona_data if hasattr(ag, 'persona_data') else None)
            for ag in self.agents
        ]

    def parley(self):
        self.turn_idx += 1

        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))

        """If at first turn, we need to give each agent their persona"""
        if self.turn_idx == 1:
            for idx, agent in enumerate(self.agents):
                persona_text = ''
                for s in self.personas[idx]:
                    persona_text += (
                        '<b><span style="color:blue">'
                        '{}\n</span></b>'.format(s.strip())
                    )
                control_msg['persona_text'] = persona_text
                control_msg['text'] = self.get_instruction(
                    tag='start', agent_id=agent.id
                )
                agent.observe(validate(control_msg))
                if idx == 0:
                    time.sleep(3)

        """If we get to the min turns, inform turker that they can end if they
           want
        """
        if self.turn_idx == self.n_turn + 1:
            for idx, agent in enumerate(self.agents):
                control_msg['text'] = self.get_instruction(idx, tag='exceed_min_turns')
                control_msg['exceed_min_turns'] = True
                agent.observe(validate(control_msg))

        """Otherwise, we proceed accordingly"""
        acts = [None, None]
        for idx, agent in enumerate(self.agents):
            if not self.chat_done:
                acts[idx] = agent.act(timeout=self.max_resp_time)
            if self.check_timeout(acts[idx]):
                return

            if self.turn_idx > 1:
                # only check if message is too short on first message
                while self.is_msg_tooshortlong(acts[idx], agent) or self.is_exact_match(
                    acts[idx], agent
                ):
                    acts[idx] = agent.act()
            else:
                while self.is_exact_match(acts[idx], agent):
                    acts[idx] = agent.act()

            if acts[idx]['episode_done']:
                self.chat_done = True
                for ag in self.agents:
                    # if agent disconnected
                    if ag != agent and ag.some_agent_disconnected:
                        control_msg['text'] = (
                            'The other worker unexpectedly diconnected. '
                            'Please click "Done with this HIT" button below to '
                            'finish this HIT.'
                        )
                        control_msg['episode_done'] = True
                        ag.observe(validate(control_msg))
                        return
                # agent ends chat after exceeding minimum number of turns
                if self.turn_idx > self.n_turn:
                    for ag in self.agents:
                        ag.observe(validate(acts[idx]))
                        control_msg['text'] = (
                            'One of you ended the chat. Thanks for your time! '
                            'Please click "Done with this HIT" button below '
                            'to finish this HIT.'
                        )
                        control_msg['episode_done'] = True
                        ag.observe(validate(control_msg))
                return

            else:
                self.dialog.append((idx, acts[idx]['text']))
                for other_agent in self.agents:
                    if other_agent != agent:
                        other_agent.observe(validate(acts[idx]))

    def shutdown(self):
        global shutdown_agent

        def shutdown_agent(mturk_agent):
            mturk_agent.shutdown()

        Parallel(n_jobs=len(self.agents), backend='threading')(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )

    def episode_done(self):
        return self.chat_done

    def get_instruction(self, agent_id=None, tag='first'):
        if tag == 'start':
            return (
                '\nSuccessfully matched. Now let\'s get to know each other'
                'through the chat! \nYou need to finish at least <b>'
                + str(self.n_turn)
                + ' chat turns</b>, after that you can click the "Done" button '
                'to end the chat. \n'
                '<b>You can track your character description on the left.</b> '
                '\n <span style="color:blue"><b>Please try to speak to the '
                'other person as if you are the character assigned.</b></span>'
                '\n <span style="color:blue"><b>Do not trivially copy the '
                'character descriptions into the message.</b></span>'
            )

        if tag == 'chat_not_done':
            return (
                'Sorry, we need at least <b>'
                + str(self.n_turn + 1 - self.turn_idx)
                + ' more turn(s)</b> to finish. '
                'Please send a new message:'
            )

        if tag == 'timeout':
            return (
                '<b>{}</b> is timeout. Please click the "Done with this HIT" '
                'button below to exit this HIT. No rejections.'.format(agent_id)
            )

        if tag == 'exceed_min_turns':
            return (
                '\n {} chat turns finished! \n Keep chatting or you can click '
                'the "Done" button to end the chat if it\'s your turn.'.format(
                    self.n_turn
                )
            )

    def save_data(self):
        # save persona_idx_stack
        convo_finished = True
        bad_workers = []
        for ag in self.agents:
            if (
                ag.hit_is_abandoned
                or ag.hit_is_returned
                or ag.disconnected
                or ag.hit_is_expired
            ):
                bad_workers.append(ag.worker_id)
                convo_finished = False
        if not convo_finished or self.dialog == []:
            for ag in self.agents:
                ag.not_approve = True
                ag.persona_generator.push_persona(ag.persona_idx)
                print(
                    "\n******* Push persona {} back to stack. *******\n".format(
                        ag.persona_idx
                    )
                )

        data_path = self.opt['extract_personas_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(
                data_path,
                '{}_{}_{}.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type,
                ),
            )
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_incomplete.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type,
                ),
            )
        print(self.world_tag + ': Data successfully saved at {}.'.format(filename))
        pickle.dump(
            {
                'personas': self.personas,
                'dialog': self.dialog,
                'workers': [ag.worker_id for ag in self.agents],
                'bad_workers': bad_workers,
                'n_turn': self.n_turn,
            },
            open(filename, 'wb'),
        )

    def is_exact_match(self, act, ag, tolerance=0):
        if act['episode_done']:
            return False

        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        text = act['text']
        if text not in ['', ' ', '  ', '   ']:
            n_word_match = 0
            for per in ag.persona_data:
                per_parse = per.split(' ')
                regular_words = ['', ' ', 'I', 'I\'m', 'My', 'i']
                for r_w in regular_words:
                    if r_w in per_parse:
                        per_parse.remove(r_w)
                per_subseq = [
                    ' '.join(per_parse[i : i + len(per_parse) - tolerance])
                    for i in range(tolerance + 1)
                ]
                for pp in per_subseq:
                    if pp in ['', ' ', '  ', '   ']:
                        per_subseq.remove(pp)
                n_word_match += sum([(paa in text) for paa in per_subseq])
            if n_word_match > 0:
                control_msg['text'] = (
                    'We found that you <b><span style="color:red">trivially '
                    'copied character descriptions</span></b>. Please '
                    'rephrase your message again.'
                )
                ag.observe(validate(control_msg))
                return True
            else:
                return False

    def is_msg_tooshortlong(self, act, ag, th_min=5, th_max=17):
        if act['episode_done']:
            return False

        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        msg_len = len(act['text'].split(' '))
        if msg_len < th_min:
            control_msg['text'] = (
                'Your message is too short, please make it more than '
                '<b><span style="color:red">5 words</span></b>.'
            )
            ag.observe(validate(control_msg))
            return True
        if msg_len > th_max:
            control_msg['text'] = (
                'Your message is too long, please make it less than '
                '<b><span style="color:red">15 words</span></b>.'
            )
            ag.observe(validate(control_msg))
            return True
        return False

    def reset_random(self):
        self.n_turn = np.random.randint(self.range_turn[0], self.range_turn[1]) + 1

    def check_timeout(self, act):
        if act['text'] == '[TIMEOUT]' and act['episode_done']:
            control_msg = {'episode_done': True}
            control_msg['id'] = 'SYSTEM'
            control_msg['text'] = self.get_instruction(
                agent_id=act['id'], tag='timeout'
            )
            for ag in self.agents:
                if ag.id != act['id']:
                    ag.observe(validate(control_msg))
            self.chat_done = True
            return True
        else:
            return False

    def review_work(self):
        global review_agent

        def review_agent(ag):
            if hasattr(ag, 'not_approve'):
                pass
            else:
                ag.approve_work()

        Parallel(n_jobs=len(self.agents), backend='threading')(
            delayed(review_agent)(agent) for agent in self.agents
        )
