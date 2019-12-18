#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.utils.safety import OffensiveStringMatcher
from parlai.core.worlds import validate, MultiAgentDialogWorld
from joblib import Parallel, delayed
from task_configs.task_config_personality import task_config as config_personality
from task_configs.task_config_caption import task_config as config_caption
from task_configs.task_config_no_personality import task_config as config_no_personality
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

TASK_TYPE_TO_CONFIG = {
    'personality': config_personality,
    'no_personality': config_no_personality,
    'caption': config_caption,
}

TASK_TYPE_TO_TITLE = {
    'personality': "Comment on an Image",
    'no_personality': "Comment on an Image",
    'caption': "Caption an Image",
}

COMMENTER = 'Commenter'
PERSONALITY_ONBOARD_MSG = '\nWelcome! Below is your personality \
        (you will be able to find it on the left side of the chat)\n \
        When you are ready to begin, \
        click the "I am ready, continue" button below\n'
ONBOARD_MSG = '\nWelcome! \
        When you are ready to begin, \
        click the "I am ready, continue" button below\n'
START_MSGS = {
    'personality': '\nImage number {}! Please take a look at the image, and \
            leave an engaging comment. \
            <b>You can see your personality on the left.</b> \n\
            <span style="color:blue"><b>Please try to comment on the image \
            as if YOU have the personality assigned.</b></span> \n',
    'no_personality': '\nImage number {}! Please take a look at the image, and \
                       write an engaging comment.',
    'caption': '\nImage number {}! Please take a look at the image, and \
                write a caption.',
}
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
CHAT_ENDED_MSG = 'You are done with {} images. Thanks for your time! \n\
        Please indicate on the left pane how much you enjoyed completing \
        this task from 1 to 5, where \
        5 is the most enjoyable and 1 is the least. Then, please explain\
        in the chat box your reason for choosing this evaluation. \
        Finally, \
        click <span style="color:blue"><b>Done with this HIT</b> </span> button \
        below to finish this HIT.'
WAITING_MSG = 'Please wait...'
OFFENSIVE_MSG = 'Our system detected that your previous comment contained \
        offensive language. Please write a different comment, thanks!'


def load_image(path):
    return Image.open(path).convert('RGB')


class PersonalityAndImageGenerator(object):
    """For the test set - specifying multiple personalities per image"""

    def __init__(self, opt):
        self.idx_stack_path = os.path.join(
            os.getcwd(),
            './personality_and_image_idx_stack{}.pkl'.format(
                '_sandbox' if opt['is_sandbox'] else ''
            ),
        )
        self.data_path = os.path.join(opt['datapath'], 'personality_captions/test.json')

        if opt.get('yfcc_path'):
            self.image_path = opt['yfcc_path']
        else:
            self.image_path = os.path.join(opt['datapath'], 'yfcc_images')
        self.personality_image_pairs = []

        with open(self.data_path) as f:
            test = json.load(f)
            self.personality_image_pairs = [
                (d['image_hash'], d['personality']) for d in test
            ]

        if os.path.exists(self.idx_stack_path):
            with open(self.idx_stack_path, 'rb') as handle:
                self.idx_stack = pickle.load(handle)
        else:
            self.idx_stack = []
            self.add_idx_stack()
            self.save_idx_stack()

    def add_idx_stack(self):
        stack = [i for i in range(len(self.personality_image_pairs))]
        random.seed()
        random.shuffle(stack)
        self.idx_stack = stack + self.idx_stack

    def pop_pair(self):
        if len(self.idx_stack) == 0:
            self.add_idx_stack()
        idx = self.idx_stack.pop()
        image_idx, personality = self.personality_image_pairs[idx]
        image_idx = int(image_idx)
        image_path = os.path.join(self.image_path, image_idx)
        return (idx, personality, image_idx, image_path)

    def push_pair(self, idx):
        self.idx_stack.append(idx)

    def save_idx_stack(self):
        with open(self.idx_stack_path, 'wb') as handle:
            pickle.dump(self.idx_stack, handle)


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


class ImageGenerator(object):
    """
    Retrieve Image from Flicker 100m set.
    """

    def __init__(self, opt):
        self.images_idx_stack_path = os.path.join(os.getcwd(), './images_idx_stack.pkl')
        if opt.get('yfcc_path'):
            self.image_path = opt['yfcc_path']
        else:
            self.image_path = os.path.join(opt['datapath'], 'yfcc_images')
        self.images = [im for im in os.listdir(self.image_path) if '.jpg' in im]

        if os.path.exists(self.images_idx_stack_path):
            with open(self.images_idx_stack_path, 'rb') as handle:
                self.idx_stack = pickle.load(handle)
        else:
            self.idx_stack = []
            self.add_idx_stack()
            self.save_idx_stack()

    def add_idx_stack(self):
        stack = self.images
        random.seed()
        random.shuffle(stack)
        self.idx_stack = stack + self.idx_stack

    def pop_image(self):
        if len(self.idx_stack) == 0:
            self.add_idx_stack()
        idx = self.idx_stack.pop()
        data = os.path.join(self.image_path, idx)
        print(data)
        return (idx, data)

    def push_image(self, idx):
        self.idx_stack.append(idx)

    def save_idx_stack(self):
        with open(self.images_idx_stack_path, 'wb') as handle:
            pickle.dump(self.idx_stack, handle)


class RoleOnboardWorld(MTurkOnboardWorld):
    """
    A world that provides a Personality to the MTurkAgent, and provides the appropriate
    instructions during onboarding.
    """

    def __init__(self, opt, mturk_agent):
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_onboard_time = opt['max_onboard_time']
        self.config = TASK_TYPE_TO_CONFIG[opt['task_type']]
        super().__init__(opt, mturk_agent)

    def parley(self):
        onboard_msg = {'id': 'SYSTEM', 'show_personality': True, 'text': ONBOARD_MSG}

        onboard_msg['task_description'] = self.config['task_description']
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


class MTurkPersonalityCaptionsWorld(MultiAgentDialogWorld):
    """
    World an agent observes ten images, with ten different personalities, and writes
    engaging comments about them.
    """

    def __init__(self, opt, agents=None, shared=None, world_tag='NONE'):
        self.turn_idx = 0
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.chat_done = False
        self.world_tag = world_tag
        self.max_resp_time = opt['max_resp_time']  # in secs
        super().__init__(opt, agents, shared)
        self.agents = agents
        self.offensive_lang_detector = OffensiveStringMatcher()
        self.agent = agents[0]
        self.data = []
        self.exact_match = False
        self.num_images = opt['num_images']
        self.multiple_personality = opt.get('multiple_personality', False)
        self.eval = 0
        self.data_type = opt['task_type']
        self.task_type_title = TASK_TYPE_TO_TITLE[opt['task_type']]
        self.config = TASK_TYPE_TO_CONFIG[opt['task_type']]

    def episode_done(self):
        return self.chat_done

    def parley(self):
        """
        COMMENTER is given an image, and is told to give a comment for the image.
        """
        # Initial Message Value
        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        # First, we give COMMENTER their personality instructions, and image
        while self.turn_idx < self.num_images:
            print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))
            # Send personality + image to turker
            if not self.multiple_personality:
                pers_tup = self.agent.personality_generator.pop_personality()
                self.pers_idx, personality = pers_tup
                img_tup = self.agent.image_generator.pop_image()
                self.image_hash, image_path = img_tup
                img = load_image(image_path)
            else:
                pair_tup = self.agent.personality_and_image_generator.pop_pair()
                self.pair_idx, personality, self.image_hash, image_path = pair_tup
                img = load_image(image_path)
            buffered = BytesIO()
            img.save(buffered, format='JPEG')
            encoded = str(base64.b64encode(buffered.getvalue()).decode('ascii'))
            control_msg['image'] = encoded
            if self.data_type == 'personality':
                personality_text = (
                    '<b><span style="color:blue">'
                    '{}\n</span></b>'.format(personality.strip())
                )
                control_msg['personality_text'] = personality_text
            control_msg['text'] = self.get_instruction(
                tag='start', agent_id=self.agent.id, turn_num=self.turn_idx + 1
            )
            control_msg['description'] = self.config['task_description']
            control_msg['task_type_title'] = self.task_type_title
            self.agent.observe(validate(control_msg))
            time.sleep(1)

            # Collect comment from turker
            offensive_counter = 0
            while offensive_counter < 3:
                idx = 0
                acts = self.acts
                acts[idx] = self.agent.act(timeout=self.max_resp_time)
                agent_left = self.check_timeout(acts[idx])
                if agent_left:
                    break
                comment = acts[idx]['text']
                offensive = self.offensive_lang_detector.contains_offensive_language(
                    comment
                )
                if offensive:
                    # Tell Turker to not be offensive!
                    offensive_msg = {'id': 'SYSTEM', 'text': OFFENSIVE_MSG}
                    self.agent.observe(validate(offensive_msg))
                    offensive_counter += 1
                else:
                    break

            if self.chat_done:
                break
            self.data.append(
                {
                    'comment': comment,
                    'personality': personality,
                    'image_hash': self.image_hash,
                    'image_path': image_path,
                    'contains_offensive_language': offensive,
                }
            )
            self.turn_idx += 1

        if self.turn_idx == self.num_images:
            control_msg['text'] = CHAT_ENDED_MSG.format(self.num_images)
            control_msg['eval'] = True
            self.agent.observe(validate(control_msg))
            act = self.agent.act(timeout=self.max_resp_time)
            self.check_timeout(act)
            try:
                ev_val = int(act['eval'])
                ev_text = act['text']
            except BaseException:  # If there is a disconnect here
                ev_val, ev_text = (-1, 'Eval not received')
            self.eval = {ev_val: ev_text}
        self.chat_done = True
        return

    def get_instruction(self, agent_id=None, tag='first', turn_num=0):
        if tag == 'start':
            return START_MSGS[self.data_type].format(turn_num)
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
            if not self.multiple_personality:
                ag.personality_generator.push_personality(self.pers_idx)
                ag.image_generator.push_image(self.image_hash)
                print(
                    '\n**Push personality {} back to stack. **\n'.format(self.pers_idx)
                )
                print('\n**Push image {} back to stack. **\n'.format(self.image_hash))
            else:
                ag.personality_and_image_generator.push_pair(self.pair_idx)
                print('\n**Push pair {} back to stack. **\n'.format(self.pair_idx))
        self.agents[0].personality_generator.save_idx_stack()
        self.agents[0].image_generator.save_idx_stack()
        self.agents[0].personality_and_image_generator.save_idx_stack()
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
        comments = [d['comment'] for d in self.data]

        if len(comments) >= 2:
            c = comments[0]
            if _exact_match(c, comments[1:]):
                self.exact_match = True

        pickle.dump(
            {
                'data': self.data,
                'worker': self.agents[0].worker_id,
                'hit_id': self.agents[0].hit_id,
                'assignment_id': self.agents[0].assignment_id,
                'exact_match': self.exact_match,
                'task_eval': self.eval,
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
                    'least one of your comments contains '
                    'offensive language'
                )
                print(
                    'Rejected work for agent {} for '
                    'offensive language'.format(ag.worker_id)
                )
            elif self.exact_match:
                ag.reject_work(
                    reason='We have rejected this HIT because all '
                    'of your comments are the exact same'
                )
                print(
                    'Rejected work for agent {} for '
                    'same comments'.format(ag.worker_id)
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
