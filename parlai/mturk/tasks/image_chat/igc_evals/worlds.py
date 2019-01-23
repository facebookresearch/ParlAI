# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.core.image_featurizers import ImageLoader
from joblib import Parallel, delayed
from task_configs.task_config_questions import task_config as config_questions
from task_configs.task_config_responses import task_config as config_responses
import numpy as np
import time
import os
import pickle
import random
import json
import base64
from io import BytesIO

RATER = 'Rater'
ONBOARD_MSG = '\nWelcome! \
        When you are ready to begin, \
        click the "I am ready, continue" button below\n'
RATE_MSG = '\nImage number {}! Please take a look at the image and textual context, and \
            rate the quality of each question from 1 to 3, \
            where 3 is the highest quality.'
RATE_RESPONSE_MSG = '\nImage number {}! Please take a look at the image, \
                    textual context, and question, \
                    and rate the quality of each response from 1 to 3, \
                    where 3 is the highest quality.'
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
CHAT_ENDED_MSG = 'You are done with {} images. Thanks for your time! \nPlease \
        click <span style="color:blue"><b>Done with this HIT</b> </span> \
        button below to finish this HIT.'
WAITING_MSG = 'Please wait...'


demo_example = {
    'questions': {
        'human': 'What is this?',
        'model': 'Is that a banana!',
        'model_sweet': 'Is that not the cutest banana?',
    },
    'responses': {
        'human': 'I think it is a banana',
        'model': 'It is a banana',
        'model_sweet': 'It is just about the cutest banana!'
    },
    'question': 'What is this?',
    'context': 'What a weird looking fruit.'
}


class IGCExampleGenerator(object):
    """Retrieve Example from Comment Battle Dataset"""
    def __init__(self, opt):
        handle = './examples_stack{}{}.pkl'.format(
            '_sandbox' if opt['is_sandbox'] else '_',
            opt['dialog_round'])
        self.examples_idx_stack_path = os.path.join(os.getcwd(), handle)
        data_path = opt.get('eval_data_path')
        if data_path != '':
            with open(data_path) as f:
                self.data = json.load(f)
        else:
            self.data = {123: demo_example}

        if os.path.exists(self.examples_idx_stack_path):
            with open(self.examples_idx_stack_path, 'rb') as handle:
                self.idx_stack = pickle.load(handle)
        else:
            self.idx_stack = []
            self.add_idx_stack()
            self.save_idx_stack()

    def add_idx_stack(self):
        stack = list(self.data.keys())
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
    """A world that provides the appropriate instructions during onboarding"""
    def __init__(self, opt, mturk_agent):
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_onboard_time = opt['max_onboard_time']
        self.round = opt['dialog_round']
        super().__init__(opt, mturk_agent)

    def parley(self):
        onboard_msg = {
            'id': 'SYSTEM',
            'text': ONBOARD_MSG}

        if self.round == 'questions':
            onboard_msg['task_description'] = config_questions['task_description']
        else:
            onboard_msg['task_description'] = config_responses['task_description']

        self.mturk_agent.observe(onboard_msg)

        act = self.mturk_agent.act(timeout=self.max_onboard_time)

        # timeout
        if act['episode_done'] or (('text' in act and
                                    act['text'] == TIMEOUT_MESSAGE)):
            self.episodeDone = True
            return

        if 'text' not in act:
            control_msg = {'id': 'SYSTEM',
                           'text': WAITING_MSG}
            self.mturk_agent.observe(validate(control_msg))
            self.episodeDone = True


class MTurkIGCEvalWorld(MultiAgentDialogWorld):
    """World where an agent observes 5 images and 3 comments about the images,
       and ranks the comments
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
        self.data = []
        self.exact_match = False
        self.num_images = opt['num_images']
        self.d_rnd = opt.get('dialog_round')
        self.image_path = opt.get('image_path')
        self.task_dir = opt['task_dir']
        opt['image_mode'] = 'raw'
        self.image_loader = ImageLoader(opt)

    def episode_done(self):
        return self.chat_done

    def parley(self):
        """RATER is given an image, context (and possibly some questions)
           and is asked to rate the responses.
        """
        # Initial Message Value
        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        """First, we give RATER the image and context
        """
        while self.turn_idx < self.num_images:
            print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))
            # Send image to turker
            if self.d_rnd == 'questions':
                control_msg['description'] = config_questions['task_description']
            else:
                control_msg['description'] = config_responses['task_description']
            self.example_id, igc_example = self.agent.example_generator.pop_example()
            img = self.image_loader.load(self.image_id_to_path(self.example_id))
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            encoded = str(base64.b64encode(buffered.getvalue()).decode('ascii'))
            control_msg['image'] = encoded
            control_msg['context'] = igc_example['context']
            """
                Setup Options for rating
            """
            if self.d_rnd == 'questions':
                options = [(k, v) for k, v in igc_example['questions'].items()]
            else:
                control_msg['question'] = igc_example['question']
                options = [(k, v) for k, v in igc_example['responses'].items()]
            random.shuffle(options)
            options, dup_dict = self.filter_option_duplicates(options)
            control_msg['options'] = [c[1] for c in options]
            # Collect rating from turker
            rate_msg = RATE_MSG if self.d_rnd == 'questions' else RATE_RESPONSE_MSG
            control_msg['text'] = rate_msg.format(self.turn_idx + 1)
            control_msg['new_eval'] = True
            self.agent.observe(validate(control_msg))
            time.sleep(1)
            act = self.agent.act(timeout=self.max_resp_time)
            # First timeout check
            self.check_timeout(act)
            if self.chat_done:
                break
            try:
                ratings = []
                collected_ratings = list(zip([q[0] for q in options], act['ratings']))
                for opt, rating in collected_ratings:
                    for other_opt in dup_dict[opt]:
                        ratings.append((other_opt, rating))
                igc_example['ratings'] = ratings
            except Exception:
                # Agent disconnected
                break
            igc_example['dialog_round_evaluated'] = self.d_rnd
            self.data.append(igc_example)
            self.turn_idx += 1

        if self.turn_idx == self.num_images:
            control_msg['text'] = CHAT_ENDED_MSG.format(self.num_images)
            self.agent.observe(validate(control_msg))
        self.chat_done = True
        return

    def image_id_to_path(self, image_id):
        if self.image_path == '':
            return os.path.join(self.task_dir, 'banana.jpg')
        else:
            return '{}/{}.jpg'.format(self.image_path, id)

    def filter_option_duplicates(self, options):
        # options = [(opt, text), (opt2, text2), ...]
        new_options = []
        text_to_opt = {}
        opt_to_opt = {}
        for opt, text in options:
            if text not in text_to_opt:
                text_to_opt[text] = opt
                new_options.append([opt, text])
                opt_to_opt[opt] = [opt]
            else:
                opt_to_opt[text_to_opt[text]].append(opt)
        return new_options, opt_to_opt

    def check_timeout(self, act):
        if act['text'] == '[TIMEOUT]' and act['episode_done']:
            control_msg = {'episode_done': True}
            control_msg['id'] = 'SYSTEM'
            control_msg['text'] = TIMEOUT_MSG
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
            if (ag.hit_is_abandoned or ag.hit_is_returned or
                    ag.disconnected or ag.hit_is_expired):
                convo_finished = False
        if not convo_finished:
            ag.example_generator.push_example(self.example_id)
            print("\n**Push image {} back to stack. **\n".format(
                    self.example_id))
        self.agents[0].example_generator.save_idx_stack()
        data_path = self.opt['data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(
                data_path,
                '{}_{}_{}.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type))
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_incomplete.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type))
        pickle.dump({'data': self.data,
                     'worker': self.agents[0].worker_id,
                     'hit_id': self.agents[0].hit_id,
                     'assignment_id': self.agents[0].assignment_id
                     }, open(filename, 'wb'))
        print('{}: Data successfully saved at {}.'.format(
            self.world_tag,
            filename))

    def review_work(self):
        global review_agent

        def review_agent(ag):
            pass  # auto approve 5 days
        Parallel(
            n_jobs=len(self.agents),
            backend='threading'
        )(delayed(review_agent)(agent) for agent in self.agents)

    def shutdown(self):
        """Shutdown all mturk agents in parallel, otherwise if one mturk agent
        is disconnected then it could prevent other mturk agents from
        completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            agent.shutdown()
        Parallel(
            n_jobs=len(self.agents),
            backend='threading'
        )(delayed(shutdown_agent)(agent) for agent in self.agents)
