#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.utils.safety import OffensiveStringMatcher
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.tasks.image_chat.build import build as build_ic
from parlai.tasks.personality_captions.build import build as build_pc
from joblib import Parallel, delayed
from task_configs.task_config_first_response import task_config as config_first
from task_configs.task_config_second_response import task_config as config_second
import numpy as np
import time
import os
import pickle
import random
import json
import base64
from parlai.core.metrics import _exact_match
from io import BytesIO
from PIL import Image

RESPONDER = 'Responder'
ONBOARD_MSG = '\nWelcome! \
        When you are ready to begin, \
        click the "I am ready, continue" button below\n'
START_MSG = '\nImage + comment number {}! Please take a look at both the image \
        and the comment, and leave an engaging response. \
        <b>You can see your personality on the left.</b> \n\
        <span style="color:blue"><b>Please try to respond to the comment \
        as if YOU have the personality assigned</b></span> - you are not \
        writing another caption.\n'
START_MSG_SECOND_RESP = '\nImage + dialog number {}! Please take a look at \
        both the image and the dialog, and leave an engaging response. \
        <b>You can see your personality on the left.</b> \n\
        <span style="color:blue"><b>Please try to respond \
        as if YOU have the personality assigned</b></span> - you are not \
        writing a caption.\n'
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
CHAT_ENDED_MSG = 'You are done with {} images. Thanks for your time! \n\
        Please click <span style="color:blue"><b>Done with this HIT</b>\
        </span> button below to finish this HIT.'
WAITING_MSG = 'Please wait...'
OFFENSIVE_MSG = 'Our system detected that your previous response contained \
        offensive language. Please write a different response, thanks!'


def load_image(path):
    return Image.open(path).convert('RGB')


class PersonalityGenerator(object):
    def __init__(self, opt):
        self.personalities_path = os.path.join(
            opt['datapath'], 'personality_captions/personalities.json'
        )
        self.personalities_idx_stack_path = os.path.join(
            os.getcwd(), './personalities_idx_stack.pkl'
        )

        self.personalities = []

        with open(self.personalities_path) as f:
            p_dict = json.load(f)
            self.personalities = [p for p_type in p_dict.values() for p in p_type]

        if os.path.exists(self.personalities_idx_stack_path):
            with open(self.personalities_idx_stack_path, 'rb') as handle:
                self.idx_stack = pickle.load(handle)
        else:
            self.idx_stack = []
            self.add_idx_stack()
            self.save_idx_stack()

    def add_idx_stack(self):
        stack = [i for i in range(len(self.personalities))]
        random.seed()
        random.shuffle(stack)
        self.idx_stack = stack + self.idx_stack

    def pop_personality(self):
        if len(self.idx_stack) == 0:
            self.add_idx_stack()
        idx = self.idx_stack.pop()
        data = self.personalities[idx]
        return (idx, data)

    def push_personality(self, idx):
        self.idx_stack.append(idx)

    def save_idx_stack(self):
        with open(self.personalities_idx_stack_path, 'wb') as handle:
            pickle.dump(self.idx_stack, handle)


class ExampleGenerator(object):
    """
    Retrieve Example from Personality-Captions.
    """

    def __init__(self, opt):
        self.second_resp = opt.get('second_response')
        self.examples_idx_stack_path = os.path.join(
            os.getcwd(),
            './{}_examples_stack{}.pkl'.format(
                'second_response' if self.second_resp else 'first_response',
                '_sandbox' if opt['is_sandbox'] else '',
            ),
        )
        self.OLD = OffensiveStringMatcher()
        self.opt = opt
        build_pc(opt)
        build_ic(opt)
        df = 'personality_captions' if not self.second_resp else 'image_chat'
        data_path = os.path.join(self.opt['datapath'], '{}/{}.json')
        self.data = []
        for dt in ['train', 'val', 'test']:
            if self.second_resp and dt == 'val':
                dt = 'valid'
            with open(data_path.format(df, dt)) as f:
                self.data += json.load(f)

        if self.second_resp:
            self.data = [d for d in self.data if len(d['dialog']) > 1]

        if os.path.exists(self.examples_idx_stack_path):
            with open(self.examples_idx_stack_path, 'rb') as handle:
                self.idx_stack = pickle.load(handle)
        else:
            self.idx_stack = []
            self.add_idx_stack()
            self.save_idx_stack()

    def add_idx_stack(self):
        stack = list(range(len(self.data)))
        random.seed()
        random.shuffle(stack)
        self.idx_stack = stack + self.idx_stack

    def pop_example(self):
        if len(self.idx_stack) == 0:
            self.add_idx_stack()
        idx = self.idx_stack.pop()
        ex = self.data[idx]
        return (idx, ex)

    def push_example(self, idx):
        self.idx_stack.append(idx)

    def save_idx_stack(self):
        with open(self.examples_idx_stack_path, 'wb') as handle:
            pickle.dump(self.idx_stack, handle)


class RoleOnboardWorld(MTurkOnboardWorld):
    """
    A world that provides the appropriate instructions during onboarding.
    """

    def __init__(self, opt, mturk_agent):
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_onboard_time = opt['max_onboard_time']
        self.second_resp = opt['second_response']
        super().__init__(opt, mturk_agent)

    def parley(self):
        onboard_msg = {'id': 'SYSTEM', 'text': ONBOARD_MSG}
        config = config_first if not self.second_resp else config_second
        onboard_msg['task_description'] = config['task_description']
        self.mturk_agent.observe(onboard_msg)

        act = self.mturk_agent.act(timeout=self.max_onboard_time)

        # timeout
        if act['episode_done'] or (('text' in act and act['text'] == TIMEOUT_MESSAGE)):
            self.episodeDone = True
            return

        if 'text' not in act:
            control_msg = {'id': 'SYSTEM', 'text': WAITING_MSG}
            self.mturk_agent.observe(validate(control_msg))
            self.episodeDone = True


class MTurkImageChatWorld(MultiAgentDialogWorld):
    """
    World where an agent observes 5 images and 5 comments, with 5 different
    personalities, and writes engaging responses to the comments.
    """

    def __init__(self, opt, agents=None, shared=None, world_tag='NONE'):
        self.turn_idx = 0
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.chat_done = False
        self.world_tag = world_tag
        self.max_resp_time = opt['max_resp_time']  # in secs
        super().__init__(opt, agents, shared)
        self.agents = agents
        self.agent = agents[0]
        self.offensive_lang_detector = OffensiveStringMatcher()
        self.data = []
        self.exact_match = False
        self.num_images = opt['num_images']
        self.second_resp = opt.get('second_response', False)
        self.config = config_first if not self.second_resp else config_second
        if opt.get('yfcc_path'):
            self.image_path = opt['yfcc_path']
        else:
            self.image_path = os.path.join(opt['datapath'], 'yfcc_images')

    def episode_done(self):
        return self.chat_done

    def parley(self):
        """
        RESPONDER is given an image and a comment, and is told to give a response for to
        the comment.
        """
        # Initial Message Value
        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        '''We only have to worry about 1 agent'''
        agent = self.agents[0]

        '''First, we give RESPONDER their personality instructions, and image
        '''
        while self.turn_idx < self.num_images:
            print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))
            # Send personality + image + comment to turker

            self.example_num, example = self.agent.example_generator.pop_example()
            control_msg['text'] = self.get_instruction(
                tag='start', agent_id=agent.id, turn_num=self.turn_idx + 1
            )

            if self.second_resp:
                control_msg['comment_text'] = (
                    '<b><span style="color:red">'
                    '{}\n</span></b>'.format(example['dialog'][0][1].strip())
                )
                control_msg['response_text'] = (
                    '<b><span style="color:blue">'
                    '{}\n</span></b>'.format(example['dialog'][1][1].strip())
                )
            else:
                control_msg['comment_text'] = (
                    '<b><span style="color:red">'
                    '{}\n</span></b>'.format(example['comment'].strip())
                )

            img = load_image(
                os.path.join(self.image_path, '{}.jpg'.format(example['image_hash']))
            )
            buffered = BytesIO()
            img.save(buffered, format='JPEG')
            encoded = str(base64.b64encode(buffered.getvalue()).decode('ascii'))
            control_msg['image'] = encoded
            if self.second_resp:
                self.pers_idx, personality = (-1, example['dialog'][0][0])
            else:
                pers_tup = self.agent.personality_generator.pop_personality()
                self.pers_idx, personality = pers_tup
            personality_text = '<b><span style="color:{}">' '{}\n</span></b>'.format(
                'blue' if not self.second_resp else 'red', personality.strip()
            )
            control_msg['personality_text'] = personality_text
            control_msg['description'] = self.config['task_description']
            agent.observe(validate(control_msg))
            time.sleep(1)

            # Collect comment from turker
            offensive_counter = 0
            while offensive_counter < 3:
                idx = 0
                acts = self.acts
                acts[idx] = agent.act(timeout=self.max_resp_time)
                agent_left = self.check_timeout(acts[idx])
                if agent_left:
                    break
                response = acts[idx]['text']
                offensive = self.offensive_lang_detector.contains_offensive_language(
                    response
                )
                if offensive:
                    # Tell Turker to not be offensive!
                    offensive_counter += 1
                    if offensive_counter == 3:
                        break
                    offensive_msg = {'id': 'SYSTEM', 'text': OFFENSIVE_MSG}
                    agent.observe(validate(offensive_msg))
                else:
                    break

            if self.chat_done:
                break
            ex_to_save = example.copy()
            key = 'second_response' if self.second_resp else 'first_response'
            ex_to_save[key] = response
            ex_to_save['{}_personality'.format(key)] = personality
            ex_to_save['contains_offensive_language'] = offensive
            self.data.append(ex_to_save)
            self.turn_idx += 1

        if self.turn_idx == self.num_images:
            control_msg['text'] = CHAT_ENDED_MSG.format(self.num_images)
            agent.observe(validate(control_msg))
        self.chat_done = True
        return

    def get_instruction(self, agent_id=None, tag='first', turn_num=0):
        if tag == 'start':
            start_msg = START_MSG if not self.second_resp else START_MSG_SECOND_RESP
            return start_msg.format(turn_num)
        if tag == 'timeout':
            return TIMEOUT_MSG

    def check_timeout(self, act):
        if act['text'] == '[TIMEOUT]' and act['episode_done']:
            control_msg = {'episode_done': True}
            control_msg['id'] = 'SYSTEM'
            control_msg['text'] = self.get_instruction(tag='timeout')
            for ag in self.agents:
                if ag.id != act['id']:
                    ag.observe(validate(control_msg))
            self.chat_done = True
            return True
        elif act['text'] == '[DISCONNECT]':
            self.chat_done = True
            return True
        else:
            return False

    def save_data(self):
        convo_finished = True
        for ag in self.agents:
            if (
                ag.hit_is_abandoned
                or ag.hit_is_returned
                or ag.disconnected
                or ag.hit_is_expired
            ):
                convo_finished = False
        if not convo_finished:
            if not self.second_resp:
                ag.personality_generator.push_personality(self.pers_idx)
            ag.example_generator.push_example(self.example_num)
            print('\n**Push personality {} back to stack. **\n'.format(self.pers_idx))
            print('\n**Push image {} back to stack. **\n'.format(self.example_num))
        if not self.second_resp:
            self.agents[0].personality_generator.save_idx_stack()
        self.agents[0].example_generator.save_idx_stack()
        data_path = self.opt['data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(
                data_path,
                '{}_{}_{}.pkl'.format(
                    time.strftime('%Y%m%d-%H%M%S'),
                    np.random.randint(0, 1000),
                    self.task_type,
                ),
            )
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_incomplete.pkl'.format(
                    time.strftime('%Y%m%d-%H%M%S'),
                    np.random.randint(0, 1000),
                    self.task_type,
                ),
            )
        key = 'second_response' if self.second_resp else 'first_response'
        responses = [d[key] for d in self.data]

        if len(responses) >= 2:
            c = responses[0]
            if _exact_match(c, responses[1:]):
                self.exact_match = True
        data_to_save = [
            d for d in self.data if d['contains_offensive_language'] is None
        ]
        pickle.dump(
            {
                'data': data_to_save,
                'worker': self.agents[0].worker_id,
                'hit_id': self.agents[0].hit_id,
                'assignment_id': self.agents[0].assignment_id,
                'exact_match': self.exact_match,
            },
            open(filename, 'wb'),
        )
        print('{}: Data successfully saved at {}.'.format(self.world_tag, filename))

    def review_work(self):
        global review_agent

        def review_agent(ag):
            contains_offense = any(d['contains_offensive_language'] for d in self.data)
            if contains_offense:
                ag.reject_work(
                    reason='We have rejected this HIT because at '
                    'least one of your comments '
                    'contains offensive language'
                )
                print(
                    'Rejected work for agent {} for offensive language'.format(
                        ag.worker_id
                    )
                )
            elif self.exact_match:
                ag.reject_work(
                    reason='We have rejected this HIT because '
                    'all of your comments are the exact same'
                )
                print(
                    'Rejected work for agent {} for same comments'.format(ag.worker_id)
                )

        Parallel(n_jobs=len(self.agents), backend='threading')(
            delayed(review_agent)(agent) for agent in self.agents
        )

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            agent.shutdown()

        Parallel(n_jobs=len(self.agents), backend='threading')(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )
