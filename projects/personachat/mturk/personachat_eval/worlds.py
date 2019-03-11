#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.agents import create_agent
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
START_MSG = '\nSuccessfully matched. \
        Now let\'s get to know each other through the chat! \n\
        You need to finish at least <b>{} chat turns</b>, \
        after which you can click the "Done" button to end the chat. \n \
        <b>You can track your character description on the left.</b> {} \n\
        <span style="color:blue"><b>Please try to speak to the other person \
        as if you are the character assigned.</b></span> \n \
        <span style="color:blue"><b>Do not trivially copy \
        the character descriptions into the message.</b></span>'
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
RETRIEVED_PASSAGES_INST_MSG = 'Please take a look at the relevant passages \
        to your left before answering'
RETRIEVED_PASSAGES_MSG = '<span style="color:blue"> - {}\
        </span>'
WAITING_MSG = 'Please wait while we match you with another worker...'


class PersonasGenerator(object):

    def __init__(self, opt):
        self.personas_idx_stack_path = os.path.join(os.getcwd(),
                                                    './personas_idx_stack.pkl')

        self.personas_path = '{}/data/personas-{}'.format(
                             os.getcwd(),
                             opt['persona_type'] +
                                'Revised' if opt['revised'] else 'Original')
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
        print ("\n******* Pop persona {} from stack. *******\n".format(idx))
        data = np.load(os.path.join(self.personas_path,
                                    self.personas_name_list[int(idx)]))
        return (idx, data)

    def push_persona(self, idx):
        self.idx_stack.append(idx)

    def save_idx_stack(self):
        with open(self.personas_idx_stack_path, 'wb') as handle:
            pickle.dump(self.idx_stack, handle)


class PersonaProfileWorld(MTurkOnboardWorld):
    """A world that provides a persona to the MTurkAgent"""

    def __init__(self, opt, mturk_agent):
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_persona_time = opt['max_persona_time']
        super().__init__(opt, mturk_agent)

    def parley(self):
        persona_idx, data = self.mturk_agent.persona_generator.pop_persona()
        model_persona_idx, model_data = self.mturk_agent.persona_generator.pop_persona()
        self.mturk_agent.persona_idx = persona_idx
        self.mturk_agent.persona_data = data
        self.mturk_agent.model_persona = [model_persona_idx, model_data]
        self.mturk_agent.persona_pair = [(persona_idx, data), (model_persona_idx, model_data)]
        persona_text = ''
        for s in data:
            persona_text += '<b><span style="color:blue">' \
                            '{}\n</span></b>'.format(s.strip())

        self.mturk_agent.observe({
            'id': 'SYSTEM',
            'show_persona': True,
            'text': ONBOARD_MSG + '<br>' + persona_text + '<br>'})

        act = self.mturk_agent.act(timeout=self.max_persona_time)

        # timeout
        if act['episode_done'] or (('text' in act and
                                    act['text'] == TIMEOUT_MESSAGE)):

            self.mturk_agent.persona_generator.push_persona(
                self.mturk_agent.persona_idx)
            self.mturk_agent.persona_generator.save_idx_stack()
            self.episodeDone = True
            return

        if 'text' not in act:
            control_msg = {'id': 'SYSTEM',
                           'text': WAITING_MSG}
            self.mturk_agent.observe(validate(control_msg))
            self.episodeDone = True

class PersonaChatEvalWorld(MultiAgentDialogWorld):

    def __init__(self, opt, agents=None, shared=None,
                 range_turn=(4, 7), max_turn=10,
                 max_resp_time=120,
                 model_agent_opt=None,
                 world_tag='NONE',
                 agent_timeout_shutdown=120):
        self.turn_idx = 0
        self.range_turn = range_turn
        self.max_turn = max_turn
        self.n_turn = np.random.randint(self.range_turn[0], self.range_turn[1]) + 1
        self.dialog = []
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.chat_done = False
        self.n_personas = []
        self.fluency_score = len(agents) * [-1]
        self.eng_score = len(agents) * [-1]
        self.consistent_score = len(agents) * [-1]
        self.persona_picked = len(agents) * [None]
        self.world_tag = world_tag

        # set up model agent
        self.model_agent = create_agent(model_agent_opt)

        # below are timeout protocols
        self.max_resp_time = max_resp_time  # in secs
        self.agent_timeout_shutdown = agent_timeout_shutdown
        super().__init__(opt, agents, shared)
        self.personas = [(ag.persona_data if hasattr(ag, 'persona_data') else None) for ag in self.agents]

    def parley(self):
        self.turn_idx += 1

        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))

        '''If at first turn, we need to give each agent their persona'''
        if self.turn_idx == 1:
            for idx, agent in enumerate(self.agents):
                persona_text = ''
                for s in self.personas[idx]:
                    persona_text += '<b><span style="color:blue">' \
                                    '{}\n</span></b>'.format(s.strip())
                control_msg['persona_text'] = persona_text
                control_msg['text'] = self.get_instruction(
                                            tag='start',
                                            agent_id=agent.id)
                agent.observe(validate(control_msg))
                if idx == 0:
                    time.sleep(3)

        '''If we get to the min turns, inform turker that they can end if they
           want
        '''
        if self.turn_idx == self.n_turn + 1:
            for idx, agent in enumerate(self.agents):
                control_msg['text'] = self.get_instruction(idx, tag='exceed_min_turns')
                control_msg['exceed_min_turns'] = True
                agent.observe(validate(control_msg))

        '''Otherwise, we proceed accordingly'''
        acts = []
        # MTurk agent turn
        idx = 0
        agent = self.agents[0] # Mturk agent
        acts.append(agent.act(timeout=self.max_resp_time))
        if acts[idx] is not None:
            if acts[idx]['text'] == 'PERSONA':
                _text = ''
                for s in agent.model_persona[1]['persona']:
                    _text += '<b><span style="color:blue">' + s.strip() + '</span></b><br>'
                control_msg['text'] = 'The model persona is: \n' + _text
                agent.observe(control_msg)
                return
            while self.is_msg_tooshortlong(acts[idx], agent) or self.is_exact_match(acts[idx], agent):
                acts[idx] = agent.act()

            if acts[idx]['episode_done']:
                self.chat_done = True
                self.check_timeout(acts[idx])
                for ag in self.agents:
                    if ag != agent and ag.some_agent_disconnected:
                        control_msg['text'] = 'The other worker unexpectedly diconnected. \n \
                            Please click <span style="color:blue"><b>Done with this HIT</b></span> button below to exit this HIT. No rejections.'
                        ag.observe(validate(control_msg))
                        return
                if self.turn_idx > self.n_turn:
                    acts = [None]

                    # Fluency Check
                    for idx, agent in enumerate(self.agents):
                        control_msg['text'] = 'Now the conversation is completed! \n Please evaluate the other person\'s \
                                               <span style="color:blue"><b>fluency</b></span> during this conversation by \
                                               <b>entering a score from [1, 2, 3, 4, 5]</b> below, \
                                               <span style="color:blue">fluency reflects whether the other people\'s words are accurate, and whether you can read it quickly and with ease.</span>\
                                               (1 means "not fluent at all" and 5 means "extremely fluent", e.g., You can enter 3 for an OK fluency)'
                        agent.observe(validate(control_msg))
                        acts[idx] = agent.act(timeout=self.max_resp_time)
                        while acts[idx]['text'] not in ['1', '2', '3', '4', '5']:
                            control_msg['text'] = "The score you entered must be in [1, 2, 3, 4, 5]. Remember to click the SEND button and not the DONE button. Please try again:"
                            agent.observe(validate(control_msg))
                            acts[idx] = agent.act(timeout=self.max_resp_time)
                        if 'text' in acts[idx] and acts[idx]['text'] in ['1', '2', '3', '4', '5']:
                            self.fluency_score[idx] = int(acts[idx]['text'])

                    # Engagingness Check
                    for idx, agent in enumerate(self.agents):
                        control_msg['text'] = 'Now please evaluate the other people\'s \
                                              <span style="color:blue"><b>engagingness DISREGARDING the fluency</b></span> \
                                              during this conversation by <b>entering a score from [1, 2, 3, 4, 5]</b> below: \
                                               (1 means "not engaging at all" and 5 means "extremely engaging", e.g., You can enter 3 for an OK dialog)'
                        agent.observe(validate(control_msg))
                        acts[idx] = agent.act(timeout=self.max_resp_time)
                        while acts[idx]['text'] not in ['1', '2', '3', '4', '5']:
                            control_msg['text'] = "The score you entered must be in [1, 2, 3, 4, 5]. Remember to click the SEND button and not the DONE button. Please try again:"
                            agent.observe(validate(control_msg))
                            acts[idx] = agent.act(timeout=self.max_resp_time)
                        if 'text' in acts[idx] and acts[idx]['text'] in ['1', '2', '3', '4', '5']:
                            self.eng_score[idx] = int(acts[idx]['text'])

                    # Check Consistency
                    for idx, agent in enumerate(self.agents):
                        control_msg['text'] = 'Now please evaluate the other people\'s \
                                              <span style="color:blue"><b>consistency of persona</b></span> \
                                              (e.g., "I have a dog" followed by "I have no pets" is not consistent)\
                                              during this conversation by <b>entering a score from [1, 2, 3, 4, 5]</b> below: \
                                               (1 means "not consistent at all" and 5 means "extremely consistent", e.g., You can enter 3 for an OK consistency)'
                        agent.observe(validate(control_msg))
                        acts[idx] = agent.act(timeout=self.max_resp_time)
                        while acts[idx]['text'] not in ['1', '2', '3', '4', '5']:
                            control_msg['text'] = "The score you entered must be in [1, 2, 3, 4, 5]. Remember to click the SEND button and not the DONE button. Please try again:"
                            agent.observe(validate(control_msg))
                            acts[idx] = agent.act(timeout=self.max_resp_time)
                        if 'text' in acts[idx] and acts[idx]['text'] in ['1', '2', '3', '4', '5']:
                            self.consistent_score[idx] = int(acts[idx]['text'])

                    # Persona Selection
                    for idx, agent in enumerate(self.agents):
                        model_idx = agent.model_persona[0]
                        self_idx = agent.persona_idx
                        false_idx_list = [x for x in range(len(agent.persona_generator.personas_name_list))]
                        false_idx_list.remove(self_idx)
                        false_idx_list.remove(model_idx)
                        false_idx = random.choice(false_idx_list)
                        false_data = np.load(os.path.join(agent.persona_generator.personas_path, agent.persona_generator.personas_name_list[false_idx]))
                        cand_text = []
                        for dt in [agent.model_persona[1], false_data]:
                            if dt == agent.model_persona[1]:
                                is_correct = True
                            else:
                                is_correct = False
                            _text = ''
                            for s in dt:
                                _text += '<b><span style="color:blue">' + s.strip() + '</span></b><br>'
                            cand_text.append((is_correct, _text))
                        random.shuffle(cand_text)

                        control_msg['text'] = 'Now we show you two personas below, please select the one that is more likely to match \
                                               with the person you just talked to, by entering 1 or 2: \n' \
                                               + '1.<br>' \
                                               + cand_text[0][1] \
                                               +'<br>' \
                                               + '2.<br>' \
                                               + cand_text[1][1]
                        agent.observe(validate(control_msg))
                        acts[idx] = agent.act(timeout=self.max_resp_time)
                        while acts[idx]['text'] not in ['1', '2']:
                            control_msg['text'] = "The Persona index you entered must be 1 or 2. Remember to click the SEND button and not the DONE button. Please try again:"
                            agent.observe(validate(control_msg))
                            acts[idx] = agent.act(timeout=self.max_resp_time)

                        if 'text' in acts[idx] and acts[idx]['text'] in ['1', '2']:
                            self.persona_picked[idx] = cand_text[int(acts[idx]['text'])-1][0]

                    for ag in self.agents:
                        ag.observe(validate(acts[idx]))
                        control_msg['text'] = 'One of you ended the chat. Thanks for your time! \nPlease click <span style="color:blue"><b>Done with this HIT</b></span> button below to finish this HIT.'
                        ag.observe(validate(control_msg))
                return

            self.dialog.append((idx, acts[idx]['text']))
            acts[idx]['eval_labels'] = ['__NULL__']
            self.model_agent.observe(acts[idx])

        # Model_agent turn
        idx = 1
        acts.append(self.model_agent.act())

        for (sb_0, sb_1) in [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'), ('i ', 'I ')]:
            acts[idx]['text'] = acts[idx]['text'].replace(sb_0, sb_1)
        acts[idx]['text'].capitalize()
        acts[idx]['id'] = 'PERSON_2'
        acts[idx]['message_id'] = acts[0]['message_id'][:-1] + '0' if acts[0]['message_id'][-1] != '0' else acts[0]['message_id'][:-1] + '1'
        self.dialog.append((idx, acts[idx]['text']))
        time.sleep(len(acts[idx]['text'].split(' ')) * 0.5)
        agent.observe(acts[idx])


    def episode_done(self):
        return self.chat_done

    def get_instruction(self, agent_id=None, tag='first'):
        if tag == 'start':
            return '\nSuccessfully matched. Now let\'s get to know each other through the chat! \n\
                    You need to finish at least <b>' + str(self.n_turn) + ' chat turns</b>, \
                    after that you can click the "Done" button to end the chat. \n \
                    <b>You can track your character description on the left.</b> \
		    \n <span style="color:blue"><b>Please try to speak to the other person as if you are the character assigned.</b></span> \n \
                    <span style="color:blue"><b>Do not trivially copy the character descriptions into the message.</b></span>'

        if tag == 'chat_not_done':
            return 'Sorry, we need at least <b>' + str(self.n_turn + 1 - self.turn_idx) + ' more turn(s)</b> to finish. ' + \
                   'Please send a new message:'

        if tag == 'timeout':
            return '<b>{}</b> is timeout. \
                    Please click the "Done with this HIT" button below to exit this HIT. No rejections.'.format(agent_id)

        if tag == 'exceed_min_turns':
            return '\n {} chat turns finished! \n Keep chatting or you can click the "Done" button to end the chat if it\'s your turn.'.format(self.n_turn)

    def save_data(self):
        # save persona_idx_stack
        convo_finished = True
        bad_workers = []
        for ag in self.agents:
            if (ag.hit_is_abandoned or ag.hit_is_returned or \
                ag.disconnected or ag.hit_is_expired):
                bad_workers.append(ag.worker_id)
                convo_finished = False
        if not convo_finished or self.dialog == [] or self.eng_score[0] == -1 or self.fluency_score[0] == -1 or self.consistent_score[0] == -1:
            for ag in self.agents:
                ag.not_approve = True
                ag.persona_generator.push_persona(ag.persona_idx)
                print ("\n******* Push persona {} back to stack. *******\n".format(ag.persona_idx))

        data_path = self.opt['data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(data_path, '{}_{}_{}.pkl'.format(time.strftime("%Y%m%d-%H%M%S"), np.random.randint(0, 1000), self.task_type))
        else:
            filename = os.path.join(data_path, '{}_{}_{}_incomplete.pkl'.format(time.strftime("%Y%m%d-%H%M%S"), np.random.randint(0, 1000), self.task_type))
        print(self.world_tag+': Data successfully saved at {}.'.format(filename))
        self.personas.append(self.agents[0].model_persona[1])
        pickle.dump({'personas': self.personas,
                     'dialog': self.dialog,
                     'workers': [ag.worker_id for ag in self.agents],
                     'bad_workers': bad_workers,
                     'n_turn': self.n_turn,
                     'fluency_score': self.fluency_score,
                     'eng_score': self.eng_score,
                     'consistent_score': self.consistent_score,
                     'persona_picked': self.persona_picked,
                     'n_personas': self.n_personas}, open(filename, 'wb'))


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
                per_subseq = [' '.join(per_parse[i:i+len(per_parse)-tolerance]) for i in range(tolerance + 1)]
                for pp in per_subseq:
                    if pp in ['', ' ', '  ', '   ']:
                        per_subseq.remove(pp)
                n_word_match += sum([(paa in text) for paa in per_subseq])
            if n_word_match > 0:
                control_msg['text'] = 'We found that you <b><span style="color:red">trivially copied character descriptions</span></b>. Please rephrase your message again.'
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
            control_msg['text'] = 'Your message is too short, please make it more than <b><span style="color:red">5 words</span></b>.'
            ag.observe(validate(control_msg))
            return True
        if msg_len > th_max:
            control_msg['text'] = 'Your message is too long, please make it less than <b><span style="color:red">15 words</span></b>.'
            ag.observe(validate(control_msg))
            return True
        return False

    def reset_random(self):
        self.n_turn = np.random.randint(self.range_turn[0], self.range_turn[1]) + 1

    def check_timeout(self, act):
        if act['text'] == '[TIMEOUT]' and act['episode_done']:
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

    def review_work(self):
        global review_agent
        def review_agent(ag):
            if hasattr(ag, 'not_approve'):
                pass
            else:
                ag.approve_work()
        Parallel(n_jobs=len(self.agents), backend='threading')(delayed(review_agent)(agent) for agent in self.agents)
