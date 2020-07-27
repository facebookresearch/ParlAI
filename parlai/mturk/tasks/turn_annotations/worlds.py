#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import time
import os
import json
import datetime
from joblib import Parallel, delayed

from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.agents import (
    TIMEOUT_MESSAGE,
    MTURK_DISCONNECT_MESSAGE,
    RETURN_MESSAGE,
)
from parlai.mturk.core.worlds import MTurkOnboardWorld

from constants import (
    AGENT_1,
    WAITING_MSG,
    ONBOARD_TASK_DATA,
    ONBOARD_CONFIG,
    ONBOARD_TRY_AGAIN,
    ONBOARD_FAIL,
    ONBOARD_SUBMIT,
    ONBOARD_SUCCESS,
    ANNOTATIONS_CONFIG,
)
from bot_agent import TurkLikeAgent
from utils import Compatibility


class TurnAnnotationsOnboardWorld(MTurkOnboardWorld):
    """
    This onboarding world displays a sample conversation with checkboxes of the same
    annotations as in the real HIT, but it is not a live conversation (all utterances
    are displayed at once).

    constants.py has the task data with correct answers in json form
    (ONBOARD_TASK_DATA). User gets to try again onboard_failures_max_allowed times and
    is soft banned if they fail more than that.
    """

    def __init__(self, opt, mturk_agent):
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_onboard_time = opt['max_onboard_time']
        self.min_correct = ONBOARD_CONFIG['min_correct']
        self.max_incorrect = ONBOARD_CONFIG['max_incorrect']
        self.onboard_failures_max_allowed = ONBOARD_CONFIG[
            'onboard_failures_max_allowed'
        ]
        self.onboard_failure_count = 0
        self.worker_answer_file = os.path.join(
            opt['onboard_worker_answer_folder'], 'worker_answers.json'
        )
        os.makedirs(opt['onboard_worker_answer_folder'], exist_ok=True)
        super().__init__(opt, mturk_agent)

    def _is_worker_disconnected(self, act):
        return 'text' in act and act['text'] in [
            MTURK_DISCONNECT_MESSAGE,
            RETURN_MESSAGE,
            TIMEOUT_MESSAGE,
        ]

    def write_worker_answers_to_file(self, worker_answers):
        now = datetime.datetime.now()
        obj = {
            'dt': datetime.datetime.strftime(now, '%Y%m%d %H:%M:%S'),
            'worker_id': self.mturk_agent.worker_id,
            'worker_answers': worker_answers,
        }

        with open(self.worker_answer_file, 'a+') as f:
            f.write(json.dumps(obj) + '\n')

    def check_onboarding_answers(self, worker_id, worker_answers):
        """
        Calculate how many correct answers the user gave.

        :param worker_answers: dict with the keys of the message index
        (1,3,5,7,9), so can index directly into the ONBOARD_TASK_DATA for the
        answers (but the frontend sends keys that are strings)
        :return: boolean as to whether the worker passed or failed the task
        """
        number_correct = 0
        number_incorrect = 0
        answer_count = sum([len(arr) for (k, arr) in worker_answers.items()])
        for m_idx_str in worker_answers:
            m = int(m_idx_str)
            correct_answers = ONBOARD_TASK_DATA[m]['answers']
            worker_answers_for_index = worker_answers[m_idx_str]
            for ans in worker_answers_for_index:
                if ans in correct_answers:
                    number_correct += 1
                else:
                    number_incorrect += 1

        assert answer_count == (number_correct + number_incorrect)
        print(
            f'Worker {worker_id} got {number_correct} annotations correct and {number_incorrect} incorrect in onboarding.'
        )
        if (
            number_correct >= self.min_correct
            and number_incorrect <= self.max_incorrect
        ):
            return True
        return False

    def parley(self):
        print(
            f'{self.__class__.__name__}: starting parley for worker_id: {self.mturk_agent.worker_id}'
        )

        onboarding_task_html = ''
        # As in the main world, render the HTML client-side b/c it's easier with
        # this legacy (non-React) task. Bad practice... TODO: change
        for idx, utt in enumerate(ONBOARD_TASK_DATA):
            if idx % 2 == 0:
                # human
                onboarding_task_html += f"""<div class="alert alert-info" style="float: right; display:table;"><span class="onboarding-text"><b>YOU:</b> {utt["text"]}</span></div>"""
            else:
                annotations_html = TurkLikeAgent.construct_annotations_html(idx)
                onboarding_task_html += f"""<div class="alert alert-warning" style="float: left; display:table; margin-top:30px"><span class="onboarding-text"><b>THEM:</b> {utt["text"]}{annotations_html}</span></div>"""
        onboarding_task_html += '<div style="clear:both;"></div>'

        self.mturk_agent.observe(
            {
                'id': 'SYSTEM',
                'text': '',
                'onboarding_html': onboarding_task_html,
                'annotations_config': ANNOTATIONS_CONFIG,
            }
        )
        act = self.mturk_agent.act(timeout=self.max_onboard_time)
        return self._handle_act(act)

    def _handle_act(self, act):
        print(
            f'{self.__class__.__name__}: got act: {act} for worker_id: {self.mturk_agent.worker_id}'
        )

        # disconnect
        if 'text' in act and act['text'] == MTURK_DISCONNECT_MESSAGE:
            print(
                f'{self.__class__.__name__}: User disconnected {self.mturk_agent.worker_id}'
            )
            self.episodeDone = True
            return MTURK_DISCONNECT_MESSAGE

        # timeout
        if 'text' in act and act['text'] == TIMEOUT_MESSAGE:
            print(
                f'{self.__class__.__name__}: User timed out {self.mturk_agent.worker_id}'
            )
            self.episodeDone = True
            return TIMEOUT_MESSAGE

        if 'text' in act and act['text'] == ONBOARD_SUBMIT:
            print(
                f'{self.__class__.__name__}: Got first onboarding task submission for worker {self.mturk_agent.worker_id}.'
            )
            worker_answers = act['onboard_submission']
            self.write_worker_answers_to_file(worker_answers)
            if self.check_onboarding_answers(
                self.mturk_agent.worker_id, worker_answers
            ):
                print(
                    f'Worker {self.mturk_agent.worker_id} successfully passed the onboard task.'
                )

                # This will end the onboarding and send them directly to the HIT
                self.episodeDone = True
                return ONBOARD_SUCCESS
            elif self.onboard_failure_count < self.onboard_failures_max_allowed:
                # User failed but give the option to try again
                print(
                    f'{self.__class__.__name__}: Worker {self.mturk_agent.worker_id} failed onboarding but failure count is less than max times allowed, so will try again. (failure count is: {self.onboard_failure_count} of max allowed {self.onboard_failures_max_allowed}). submission is: {act["onboard_submission"]}'
                )
                self.onboard_failure_count += 1
                self.mturk_agent.observe({'id': 'System', 'text': ONBOARD_TRY_AGAIN})
                act = self.mturk_agent.act(timeout=self.max_onboard_time)
                return self._handle_act(act)
            else:
                # User has now failed too many times - soft ban them
                print(
                    f'{self.__class__.__name__}: Worker FAILED onboarding too many times...Soft blocking {self.mturk_agent.worker_id}. Submission was: {act["onboard_submission"]}'
                )
                self.mturk_agent.mturk_manager.soft_block_worker(
                    self.mturk_agent.worker_id
                )
                self.mturk_agent.observe({'id': 'System', 'text': ONBOARD_FAIL})

                # After soft ban, we just block in while loop until worker goes
                # away (Disconnects or returns the HIT as asked on the frontend)
                # If you don't do this, for some reason it still would let the
                # worker through to actual chat/HIT after onboard world shutdown
                after_block_act = self.mturk_agent.act()
                while not self._is_worker_disconnected(after_block_act):
                    self.mturk_agent.observe({'id': 'System', 'text': ONBOARD_FAIL})
                    after_block_act = self.mturk_agent.act()
                print(
                    f'{self.__class__.__name__}: Failed onboarding worker {self.mturk_agent.worker_id} did disconnect or return HIT. Ending onboarding.'
                )
                self.episodeDone = True
                return ONBOARD_FAIL

        if 'text' not in act:
            # We think that when we get here it's b/c the user has passed
            # onboarding successfully and clicked Continue to HIT button (which
            # only appears after they successfully complete the onboarding)
            print(
                f'{self.__class__.__name__}: No text in act from onboarding. Marking episode done; act was: {act}'
            )
            control_msg = {'id': 'SYSTEM', 'text': WAITING_MSG}
            self.mturk_agent.observe(validate(control_msg))
            self.episodeDone = True
            return None


class TurnAnnotationsChatWorld(MultiAgentDialogWorld):
    def __init__(
        self,
        opt,
        agents=None,
        shared=None,
        num_turns=6,
        tag=None,
        max_resp_time=120,
        agent_timeout_shutdown=120,
        annotations_config: dict = None,
    ):
        # 6 turns for a single side (so 12 total), and really it appears to be
        # 14 total b/c of the "Hi!" and first bot utterance

        self.agents = agents
        self.task_turn_idx = 0
        self.num_turns = num_turns

        self.dialog = []
        self.tag = tag
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.chat_done = False
        self.annotations_config = annotations_config

        # below are timeout protocols
        self.max_resp_time = max_resp_time  # in secs
        self.agent_timeout_shutdown = agent_timeout_shutdown
        print(
            f'Creating {self.__class__.__name__} for tag {tag} with {num_turns} turns.'
        )
        super().__init__(opt, agents, shared)

    def __add_problem_data_to_utterance(self, p):
        # Human has just responded. Problem data received
        # now will be from bot's prior utterance (turn_idx
        # is a also present to be safe that data matches)
        annotations = []
        for a in ANNOTATIONS_CONFIG:
            annotations.append(p[a['value']])
        assert any(annotations)
        self.dialog[p['turn_idx']]['problem_data'] = p

    def parley(self):
        control_msg = {
            'episode_done': False,
            'config': {
                'min_num_turns': self.num_turns,
                'annotations_config': ANNOTATIONS_CONFIG,
            },
            'left_pane_text': self.opt['left_pane_text'],
        }

        print(
            f'{self.__class__.__name__}:{self.tag}: is at turn {self.task_turn_idx} of {self.num_turns}...'
        )

        if self.task_turn_idx == 0:
            print('[Displaying "Hi!" only as per Meena task.]')
            human_first_msg = {
                'left_pane_text': self.opt['left_pane_text'],
                'episode_done': False,
                'id': self.agents[0].id,
                'text': 'Hi!',
                'fake_start': True,
                'agent_idx': 0,
            }
            for k, v in control_msg.items():
                human_first_msg[k] = v

            self.dialog.append(human_first_msg)
            self.agents[0].observe(validate(human_first_msg))
            self.agents[1].observe(validate(human_first_msg))

            first_bot_act = self.agents[1].act()
            first_bot_act = Compatibility.maybe_fix_act(first_bot_act)

            self.agents[0].observe(validate(first_bot_act))

            bot_utterance_data = {
                'agent_idx': 1,
                # Get rid of annotations HTML from bot response
                'text': first_bot_act['text'].split('<br>')[0],
                'id': first_bot_act['id'],
            }
            self.dialog.append(bot_utterance_data)
            self.task_turn_idx += 1
            return

        """Otherwise, we proceed accordingly"""
        print(
            f'{self.__class__.__name__}:{self.tag}: About to act with task turn idx: {self.task_turn_idx}'
        )
        acts = [None, None]
        for idx, agent in enumerate(self.agents):
            if not self.chat_done:
                acts[idx] = agent.act(timeout=self.max_resp_time)
                acts[idx] = Compatibility.maybe_fix_act(acts[idx])
                print(
                    f'Got act for agent idx {idx}, act was: {acts[idx]} and self.task_turn_idx: {self.task_turn_idx}.'
                )

            if self.check_timeout(acts[idx]):
                return

            if acts[idx]['episode_done']:
                self.chat_done = True
                for ag in self.agents:
                    # if agent disconnected
                    if ag != agent and ag.some_agent_disconnected:
                        if idx == 0:
                            # Human
                            control_msg['text'] = (
                                'The other worker unexpectedly diconnected. '
                                'Please click "Done with this HIT" button below to finish this HIT.'
                            )
                            control_msg['episode_done'] = True
                            ag.observe(validate(control_msg))
                        return
                # agent ends chat after exceeding minimum number of turns
                if self.task_turn_idx > self.num_turns:
                    for ag in self.agents:
                        if idx == 0:
                            print('One of you ended the chat utterance coming.')
                            control_msg['text'] = (
                                'One of you ended the chat. Thanks for your '
                                'time! Please click "Done with this HIT"'
                                'button below to finish this HIT.'
                            )
                            control_msg['episode_done'] = True
                            ag.observe(validate(control_msg))
                            # Human has just responded. Problem data received
                            # now will be from bot's prior utterance (turn_idx
                            # is a also present to be safe that data matches)
                            p = acts[idx]['problem_data_for_prior_message']
                            self.__add_problem_data_to_utterance(p)
                return

            else:
                utterance_data = {
                    'agent_idx': idx,
                    # Get rid of annotations HTML if it's the bot response
                    'text': acts[idx]['text'].split('<br>')[0],
                    'id': acts[idx]['id']
                    if 'id' in acts[idx]
                    else 'NULL_ID',  # Person1 or Polyencoder
                }
                self.dialog.append(utterance_data)
                if idx == 0:
                    # Human has just responded. Problem data received
                    # now will be from bot's prior utterance (turn_idx
                    # is a also present to be safe that data matches)
                    p = acts[idx]['problem_data_for_prior_message']
                    self.__add_problem_data_to_utterance(p)

                for other_agent in self.agents:
                    if other_agent != agent:
                        other_agent.observe(validate(acts[idx]))

                print(
                    f'[agent {idx}] self.task_turn_idx: {self.task_turn_idx}, self.dialog is: {self.dialog}'
                )
                self.task_turn_idx += 1

    def shutdown(self):
        global shutdown_agent

        def shutdown_agent(mturk_agent):
            mturk_agent.shutdown()

        Parallel(n_jobs=len(self.agents), backend='threading')(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )

    def episode_done(self):
        return self.chat_done

    def save_data(self):
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
                ag.not_approve = True

        time_string = time.strftime('%Y%m%d_%H%M%S')
        data_path = self.opt['save_folder']
        if convo_finished:
            filename = os.path.join(
                data_path,
                '{}_{}_{}.json'.format(
                    time_string, np.random.randint(0, 1000), self.task_type
                ),
            )
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_incomplete.json'.format(
                    time_string, np.random.randint(0, 1000), self.task_type
                ),
            )
        with open(os.path.join(filename), 'w+') as f_json:
            data = {
                'dialog': self.dialog,
                'workers': [ag.worker_id for ag in self.agents],
                'bad_workers': bad_workers,
                'hit_ids': [ag.hit_id for ag in self.agents],
                'assignment_ids': [ag.assignment_id for ag in self.agents],
                'task_description': {
                    'annotations_config': self.annotations_config,
                    'had_onboarding': False,
                    'model_nickname': self.agents[1].worker_id,
                    'model_file': self.agents[1].model_agent.opt['model_file'],
                    'model_opt': self.agents[1].model_agent.opt,
                },
            }
            data_str = json.dumps(data)
            f_json.write(data_str)
        print(
            f'{self.__class__.__name__}:{self.tag}: Data successfully saved at {filename} for model: {self.agents[1].worker_id}.'
        )
        return (
            self.agents[1].worker_id,
            convo_finished,
        )

    def check_timeout(self, act):
        if act['text'] == '[TIMEOUT]' and act['episode_done']:
            control_msg = {'episode_done': True}
            control_msg['id'] = 'SYSTEM'
            control_msg[
                'text'
            ] = 'HIT has timed out. Please click the "Done with this HIT" button below to exit this HIT. No rejections.'
            for ag in self.agents:
                if ag.id != act['id']:
                    if ag.id != AGENT_1:
                        ag.observe(validate(control_msg))
            self.chat_done = True
            return True
        else:
            return False

    def review_work(self):
        pass
