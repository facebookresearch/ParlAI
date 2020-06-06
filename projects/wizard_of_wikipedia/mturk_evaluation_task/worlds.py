#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.agents import create_agent_from_shared
from parlai.core.message import Message
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
import parlai.mturk.core.mturk_utils as mutils
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.tasks.wizard_of_wikipedia.build import build
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE


from joblib import Parallel, delayed
import json
import numpy as np
import os
import pickle
import random
import time


# INSTRUCTIONS
START_MSG = '\nSuccessfully matched. \
        Now let\'s talk about something through the chat!\
        <b><span style="color:red">You can find your chosen topic in the \
        bottom left corner.</span></b> \n \
        You need to finish at least <b>{} chat turns</b>, \
        after which you can click the "Done" button to end the chat. '
CHAT_NOT_DONE_MSG = 'Sorry, we need at least <b>{} more turn(s)</b> to finish. \
       Please send a new message:'
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
EXCEED_MIN_TURNS_MSG = '\n {} chat turns finished! \n \
        You can click the "Done" button to end the chat if it\'s your turn \
        or keep chatting.'
UNEXPECTED_DISCONNECTION_MSG = 'The other worker unexpectedly diconnected. \n \
        Please click <span style="color:blue"><b>Done with this HIT</b>\
        </span> button below to finish this HIT.'
CHAT_ENDED_MSG = 'One of you ended the chat. Thanks for your time! \n\
        Please click <span style="color:blue"><b>Done with this HIT</b>\
        </span> button below to finish this HIT.'
WAITING_MSG = 'Please wait while we match you with another worker...'
NAN_MSG = 'The score you entered must be in [1, 2, 3, 4, 5]. Please \
        try again:'
TOO_SHORT_MSG = 'Your message is too short, please make it more than \
        <b><span style="color:red">{} words</span></b>.'
TOO_LONG_MSG = 'Your message is too long, please make it less than \
        <b><span style="color:red">{} words</span></b>.'

# CHOOSING A TOPIC
PICK_TOPIC_MSG = 'To start, please select a topic on the left, then click the \
    \'Pick Topic\' button.'
AFTER_PICK_TOPIC_MSG = 'Thank you for selecting a topic! Now, begin the \
    conversation with your partner about the topic.'
PLEASE_WAIT_MSG = 'Your partner will choose a discussion topic. Click the \
    button below when you are ready to continue.'

# EVALUATION
OTHER_AGENT_FINISHED_MSG = '<b><span style="color:red">This chat is \
    done!</span></b> Please click \
    <span style="color:blue"><b>Done with this HIT</b></span> button below \
    to finish this HIT.'
GMARK_MSG = 'Now the conversation is completed! \n Please evaluate the \
    conversation by <b>clicking a button with score from [1, 2, 3, 4, 5]</b> \
    below, <span style="color:blue">this score should reflect how you liked \
    this conversation (1 means you did not like it at all, and 5 means it was  \
    an engaging conversation).'


class TopicsGenerator(object):
    def __init__(self, opt):
        self.opt = opt
        self.topics_path = os.path.join(
            opt['datapath'], 'wizard_of_wikipedia/topic_splits.json'
        )
        self.load_topics()

    def load_topics(self):
        if not os.path.isfile(self.topics_path):
            # download the wizard of wikipedia data
            build(self.opt)
        with open(self.topics_path) as f:
            self.data = json.load(f)
        self.seen_topics = self.data['train']
        self.unseen_topics = self.data['valid'] + self.data['test']

    def get_topics(self, seen=True, num=3):
        if seen:
            return random.sample(self.seen_topics, num)
        return random.sample(self.unseen_topics, num)


class TopicChooseWorld(MTurkOnboardWorld):
    """
    A world that provides topics to an MTurk Agent and asks them to choose one.
    """

    def __init__(self, opt, mturk_agent, role='PERSON_1'):
        self.role = role
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_choice_time = opt['max_choice_time']
        super().__init__(opt, mturk_agent)

    def parley(self):
        # only pick topic if your ID is PERSON_1
        if self.role == 'PERSON_1':
            seen = random.choice([True, False])
            num = random.choice([2, 3])
            topics = self.mturk_agent.topics_generator.get_topics(seen=seen, num=num)
            self.mturk_agent.observe(
                validate(
                    {'id': 'SYSTEM', 'text': PICK_TOPIC_MSG, 'relevant_topics': topics}
                )
            )

            topic_act = self.mturk_agent.act(timeout=self.max_choice_time)
            timed_out = self.check_timeout(topic_act)
            if timed_out:
                return
            pick_msg = AFTER_PICK_TOPIC_MSG
            self.mturk_agent.observe({'id': 'SYSTEM', 'text': pick_msg})
            self.mturk_agent.chosen_topic = topic_act['text']
            self.mturk_agent.topic_choices = topics
            self.mturk_agent.seen = seen
            self.mturk_agent.observe(validate({'id': 'SYSTEM', 'wait_msg': True}))
        else:
            self.mturk_agent.observe(validate({'id': 'SYSTEM', 'wait_msg': True}))
        return

    def check_timeout(self, act):
        if (
            (act['text'] == '[TIMEOUT]')
            or (act['text'] == '[RETURNED]')
            or (act['text'] == '[DISCONNECT]')
        ):
            return True
        else:
            return False


class WizardEval(MultiAgentDialogWorld):
    def __init__(
        self,
        opt,
        agents=None,
        shared=None,
        range_turn=(3, 5),
        max_turn=5,
        max_resp_time=120,
        model_agent_opt=None,
        world_tag='',
        agent_timeout_shutdown=120,
        knowledge_retriever_opt=None,
    ):
        self.opt = opt
        # TURN CONTROL
        self.turn_idx = 0
        self.range_turn = range_turn
        self.max_turn = max_turn
        self.n_turn = np.random.randint(self.range_turn[0], self.range_turn[1]) + 1
        self.chat_done = False
        self.other_first = random.choice([True, False])

        # DATA
        self.dialog = []
        self.dialog_list = []
        self.gmark_score = -1
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.world_tag = world_tag
        self.ratings = ['1', '2', '3', '4', '5']

        super().__init__(opt, agents, shared)

        # MODEL AGENT SET UP
        if model_agent_opt is not None:
            self.model_agent = create_agent_from_shared(model_agent_opt)
            self.knowledge_agent = create_agent_from_shared(knowledge_retriever_opt)
        else:
            # case where we test against a human
            self.model_agent = None

        # TIMEOUT PROTOCOLS
        self.max_resp_time = max_resp_time  # in secs
        self.agent_timeout_shutdown = agent_timeout_shutdown

        # TOPIC CHOICES
        if self.model_agent is None:
            for idx in range(len(self.agents)):
                if self.agents[idx].id == 'PERSON_1':
                    self.eval_agent = self.agents[idx]
                    self.other_agent = self.agents[idx - 1]
                    break
        else:
            self.eval_agent = self.agents[0]

        self.chosen_topic = self.eval_agent.chosen_topic
        self.seen = self.eval_agent.seen
        self.topic_choices = self.eval_agent.topic_choices

    def get_human_agent_act(self, agent):
        act = agent.act(timeout=self.max_resp_time)
        while self.is_msg_tooshortlong(act, agent):
            act = agent.act(timeout=self.max_resp_time)
        return act

    def _add_knowledge_to_act(self, act):
        self.knowledge_agent.observe(act, actor_id='apprentice')
        knowledge_act = self.knowledge_agent.act()
        act['knowledge'] = knowledge_act['text']
        act['checked_sentence'] = knowledge_act['checked_sentence']
        print(
            '[ Using chosen sentence from Wikpedia ]: {}'.format(
                knowledge_act['checked_sentence']
            )
        )
        act['title'] = knowledge_act['title']
        if self.opt.get('prepend_gold_knowledge', False):
            knowledge_text = ' '.join(
                [
                    TOKEN_KNOWLEDGE,
                    knowledge_act['checked_sentence'],
                    TOKEN_END_KNOWLEDGE,
                ]
            )
            new_text = '\n'.join([knowledge_text, act['text']])
            if isinstance(act, Message):
                act.force_set('text', new_text)
            else:
                act['text'] = new_text
        return act

    def parley(self):
        self.turn_idx += 1

        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))

        """If at first turn, we tell each agent the chosen topic."""
        if self.turn_idx == 1:
            self.start_time = time.time()
            for idx, agent in enumerate(self.agents):
                chosen_topic_text = (
                    '<b><span style="color:blue">'
                    '{}\n</span></b>'.format(self.chosen_topic.strip())
                )
                control_msg['chosen_topic'] = chosen_topic_text
                print(chosen_topic_text)
                control_msg['text'] = self.get_instruction(
                    tag='start', agent_id=agent.id
                )
                agent.observe(validate(control_msg))
                if idx == 0:
                    time.sleep(3)

        """If we get to the min turns, inform turker that they can end if they
        want.
        """
        if self.turn_idx == self.n_turn + 1:
            for idx, agent in enumerate(self.agents):
                control_msg['text'] = self.get_instruction(idx, tag='exceed_min_turns')
                control_msg['exceed_min_turns'] = True
                agent.observe(validate(control_msg))

        """Otherwise, we proceed accordingly."""
        # Other agent first
        if self.other_first and self.turn_idx == 1:
            if self.model_agent is not None:
                # Model has to observe chosen topic
                chosen_act = {
                    'chosen_topic': self.chosen_topic,
                    'text': self.chosen_topic,
                    'episode_done': False,
                }
                chosen_act = self._add_knowledge_to_act(chosen_act)
                self.model_agent.observe(chosen_act)
                model_act = self.model_agent.act()
                model_act.force_set('id', 'PERSON_2')
                self.dialog.append((1, model_act.get('text')))
                self.eval_agent.observe(model_act)
                self.knowledge_agent.observe(model_act, actor_id='wizard')
            else:
                act = self.get_human_agent_act(self.other_agent)
                timeout = self.check_timeout(act)
                if timeout:
                    # eval agent early disconnect
                    control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                    self.eval_agent.observe(validate(control_msg))
                    return
                else:
                    self.dialog.append((1, act.get('text')))
                    self.eval_agent.observe(act)

        # Eval agent turn
        act = self.get_human_agent_act(self.eval_agent)
        timeout = self.check_timeout(act)
        if timeout:
            if self.model_agent is None:
                control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                self.other_agent.observe(validate(control_msg))
            return

        if act['episode_done']:
            if self.turn_idx >= self.n_turn:
                self.parallel_eval_mode()

                self.chat_done = True
                for ag in self.agents:
                    control_msg['text'] = CHAT_ENDED_MSG
                    ag.observe(validate(control_msg))
            return

        self.dialog.append((0, act['text']))

        # Add chosen topic for model to observe
        act['chosen_topic'] = self.chosen_topic
        if self.model_agent is not None:
            act = self._add_knowledge_to_act(act)
            self.model_agent.observe(act)
        else:
            self.other_agent.observe(act)

        # Model_agent turn
        if not self.other_first or self.turn_idx < self.n_turn:
            if self.model_agent is not None:
                act = self.model_agent.act()
                self.knowledge_agent.observe(act, actor_id='wizard')
                text = act['text']
                for (sb_0, sb_1) in [
                    (' .', '.'),
                    (' ,', ','),
                    (' ?', '?'),
                    (' !', '!'),
                    ('i ', 'I '),
                ]:
                    text = text.replace(sb_0, sb_1)
                act.force_set('text', text)
                act.force_set('id', 'PERSON_2')
                # NOTE: your model may or may not need to observe itself here
                # If it does, call model_observes_itself or some other specialized
                # function
            else:
                act = self.get_human_agent_act(self.other_agent)
                timeout = self.check_timeout(act)
                if timeout:
                    # eval agent early disconnect
                    control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                    self.eval_agent.observe(validate(control_msg))
                    return

            self.dialog.append((1, act.get('text')))
            self.eval_agent.observe(act)

    def parallel_eval_mode(self):
        """
        Parallel function that shuts one agent down and asks the other to do the
        evaluation if their are two agents.

        If there is only one agent, it performs the evaluation.
        """
        global eval_or_shutdown

        def eval_or_shutdown(agent):
            if self.model_agent is None and agent == self.other_agent:
                control_msg = {'episode_done': False}
                control_msg['id'] = 'SYSTEM'
                control_msg['text'] = OTHER_AGENT_FINISHED_MSG
                self.other_agent.observe(validate(control_msg))
                # mark eval agent done
                self.eval_agent.mturk_manager.mark_workers_done([self.eval_agent])
                # shutdown other agent
                self.other_agent.shutdown()
            else:
                control_msg = {'episode_done': False}
                control_msg['id'] = 'SYSTEM'
                # General mark for this convo
                control_msg['text'] = GMARK_MSG
                control_msg['general_mark_score'] = True
                self.eval_agent.observe(validate(control_msg))
                act = self.eval_agent.act(timeout=self.max_resp_time)
                timeout = self.check_timeout(act)
                if timeout:
                    return
                while act['text'] not in self.ratings:
                    control_msg['text'] = NAN_MSG
                    self.eval_agent.observe(validate(control_msg))
                    act = self.eval_agent.act(timeout=self.max_resp_time)
                if 'text' in act and act['text'] in self.ratings:
                    self.gmark_score = int(act['text'])

        Parallel(n_jobs=len(self.agents), backend='threading')(
            delayed(eval_or_shutdown)(agent) for agent in self.agents
        )

    def model_observes_itself(self, txt):
        act = {'text': txt, 'episode_done': False}
        self.model_agent.observe(act)

    def episode_done(self):
        return self.chat_done

    def get_instruction(self, agent_id=None, tag='first'):
        if tag == 'start':
            return START_MSG.format(self.n_turn)

        if tag == 'chat_not_done':
            return CHAT_NOT_DONE_MSG.format(self.n_turn + 1 - self.turn_idx)

        if tag == 'timeout':
            return TIMEOUT_MESSAGE

        if tag == 'exceed_min_turns':
            return EXCEED_MIN_TURNS_MSG.format(self.n_turn)

    def save_data(self):
        convo_finished = True
        bad_workers = []
        if not self.opt['is_sandbox']:
            if self.opt.get('unique_workers') and self.opt.get('unique_qualif_id'):
                # assign qualification to evaluating agent only
                qual = self.opt['unique_qualif_id']
                mutils.give_worker_qualification(
                    self.eval_agent.worker_id, qual, value=None, is_sandbox=False
                )
        if self.dialog == [] or self.gmark_score == -1:
            convo_finished = False

        data_path = self.opt['data_path']
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
        print(self.world_tag, ': Data successfully saved at {}.'.format(filename))
        pickle.dump(
            {
                'chosen_topic': self.chosen_topic,
                'topic_choices': self.topic_choices,
                'seen': self.seen,
                'dialog': self.dialog,
                'dialog_list': self.dialog_list,
                'other_first': self.other_first,
                'total_time': time.time() - self.start_time,
                'workers': [ag.worker_id for ag in self.agents],
                'hit_id': [ag.hit_id for ag in self.agents],
                'assignment_id': [ag.assignment_id for ag in self.agents],
                'bad_workers': bad_workers,
                'n_turn': self.n_turn,
                'gmark_score': self.gmark_score,
                'inference': 'nucleus',
            },
            open(filename, 'wb'),
        )

    def is_msg_tooshortlong(self, act, ag, th_min=3, th_max=20):
        if act['episode_done']:
            return False

        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        msg_len = len(act['text'].split(' '))
        if msg_len < th_min:
            control_msg['text'] = TOO_SHORT_MSG.format(th_min)
            ag.observe(validate(control_msg))
            return True
        if msg_len > th_max:
            control_msg['text'] = TOO_LONG_MSG.format(th_max)
            ag.observe(validate(control_msg))
            return True
        return False

    def check_timeout(self, act):
        if act is None:
            self.chat_done = True
            return True
        if (
            (act['text'] == '[TIMEOUT]')
            or (act['text'] == '[RETURNED]')
            or (act['text'] == '[DISCONNECT]')
        ):
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

    def shutdown(self):
        # only need to shut down evaluating agent
        # if more than one agent, other agent shut down previously
        self.eval_agent.shutdown()
