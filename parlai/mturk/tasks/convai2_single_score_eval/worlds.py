# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from joblib import Parallel, delayed
import numpy as np
import time
import os
import pickle
import gc
import logging


# Instruction messages
ONBOARD_MSG = '\nWelcome! Below is your persona \
        (you can find it on the left side of the chat)\n \
        When you are ready to start your conversation, \
        click the "I am ready, continue" button below\n'
START_MSG = '\nSuccessfully matched. \
        Now let\'s get to know each other through the chat! \n\
        You need to finish at least <b>{} chat turns</b>, \
        after which you can click the "Done" button to end the chat. \n \
        <b>You can track your character description on the left.</b> \n\
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
WAITING_MSG = 'Please wait while we match you with another worker...'
NAN_MSG = 'The score you entered must be in [1, 2, 3, 4, 5]. Please \
        try again:'
NAN_PERSONA_MSG = 'The score you entered must be in [1, 2]. Remember to \
        click the <b>SEND</b> button and not the <b>DONE</b> button. Please \
        try again:'
TOO_SHORT_MSG = 'Your message is too short, please make it more than \
        <b><span style="color:red">{} words</span></b>.'
TOO_LONG_MSG = 'Your message is too long, please make it less than \
        <b><span style="color:red">{} words</span></b>.'
COPIED_CHARACTER_MSG = 'We found that you <b><span style="color:red">trivially \
        copied character descriptions</span></b>. Please rephrase your \
        message again.'

GMARK_MSG = 'Now the conversation is completed! \n Please evaluate the \
        conversation by <b>clicking a button with score from [1, 2, 3, 4, 5]</b> \
        below, <span style="color:blue">this score should reflect how you liked \
        this conversation (1 means you did not like it at all, and 5 means it was  \
        an engaging conversation).'
GOOD_ROUND_MSG = 'Now please select every interaction pair which you consider as a <sp\
        an style="color:blue"><b>good</b></span>, <b>natural</b> pair of messages. \
        Do not compare them between each other, try to use your life experience now.'

BAD_ROUND_MSG = 'Now please select every interaction pair which is <span style="col\
        or:blue"><b>bad</b></span>, some examples of bad partner response are: <b>n\
        ot</b> answering <b>your</b> question, answering <b>different</b> question,\
        <b>random</b> content, <b>contradicts</b> previous statements etc.'


class Convai2GeneralEval(MultiAgentDialogWorld):
    def __init__(self, opt, agents=None, shared=None,
                 range_turn=[5, 6], max_turn=10,
                 max_resp_time=120,
                 model_agent_opt=None,
                 world_tag='',
                 agent_timeout_shutdown=120):
        self.turn_idx = 0
        self.hit_id = None
        self.range_turn = range_turn
        self.max_turn = max_turn
        self.n_turn = np.random.randint(
            self.range_turn[0],
            self.range_turn[1]
        ) + 1
        self.model_name = opt.get('model_name')
        self.dialog = []

        self.reranked_cands = []
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.chat_done = False
        self.n_personas = []
        self.gmark_score = len(agents) * [-1]
        self.good_rounds = []
        self.bad_rounds = []

        self.world_tag = world_tag
        self.ratings = ['1', '2', '3', '4', '5']
        super().__init__(opt, agents, shared)

        # set up model agent
        if model_agent_opt is not None:
            self.model_agent = create_agent_from_shared(model_agent_opt)
        else:
            # case where we test against a human
            self.model_agent = self.agents[1]

        # below are timeout protocols
        self.max_resp_time = max_resp_time  # in secs
        self.agent_timeout_shutdown = agent_timeout_shutdown

        # set up personas
        self.personas = [(ag.persona_data if hasattr(ag, 'persona_data')
                          else None) for ag in self.agents]
        self.model_persona_text = '\n'.join([
            'your persona:' + pers for pers in self.agents[0].model_persona[1]
        ])
        logging.info('bot persona: {}'.format(self.model_persona_text))

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
                    persona_text += '<b><span style="color:blue">' \
                                    '{}\n</span></b>'.format(s.strip())
                control_msg['persona_text'] = persona_text
                control_msg['text'] = self.get_instruction(
                    tag='start',
                    agent_id=agent.id)
                agent.observe(validate(control_msg))
                if idx == 0:
                    time.sleep(3)

        """If we get to the min turns, inform turker that they can end if they
        want
        """
        if self.turn_idx == self.n_turn:
            for idx, agent in enumerate(self.agents):
                control_msg['text'] = self.get_instruction(
                    idx,
                    tag='exceed_min_turns'
                )
                control_msg['exceed_min_turns'] = True
                agent.observe(validate(control_msg))

        """Otherwise, we proceed accordingly"""
        acts = []
        # MTurk evaluating agent turn
        idx = 0
        agent = self.agents[0]
        acts.append(agent.act(timeout=self.max_resp_time))
        if acts[idx] is not None:
            if acts[idx]['text'] == 'PERSONA':
                _text = ''
                for s in agent.model_persona[1]['persona']:
                    _text += '<b><span style="color:blue">' + s.strip() + \
                        '</span></b><br>'
                control_msg['text'] = 'The model persona is: \n' + _text
                agent.observe(control_msg)
                return
            while self.is_msg_tooshortlong(acts[idx], agent) or \
                    self.is_exact_match(acts[idx], agent):
                acts[idx] = agent.act()

            if acts[idx]['episode_done']:
                print("Finished chat")
                self.check_timeout(acts[idx])
                for ag in self.agents:
                    if ag != agent and ag.some_agent_disconnected:
                        control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                        ag.observe(validate(control_msg))
                        return
                if self.turn_idx >= self.n_turn:
                    acts = [None]

                    # General mark for this convo
                    for idx, agent in enumerate(self.agents):
                        control_msg['text'] = GMARK_MSG
                        control_msg['general_mark_score'] = True
                        agent.observe(validate(control_msg))
                        acts[idx] = agent.act(timeout=self.max_resp_time)
                        while acts[idx]['text'] not in self.ratings:
                            control_msg['text'] = NAN_MSG
                            agent.observe(validate(control_msg))
                            acts[idx] = agent.act(timeout=self.max_resp_time)
                        if 'text' in acts[idx] and \
                                acts[idx]['text'] in self.ratings:
                            self.gmark_score[idx] = int(acts[idx]['text'])

                    # Good rounds selection
                    for idx, agent in enumerate(self.agents):
                        control_msg['text'] = GOOD_ROUND_MSG
                        control_msg['good_rounds'] = True
                        control_msg['rounds'] = '</ROUND>'.join([
                            '\n'.join(
                                [self.dialog[i][1], self.dialog[i + 1][1]])
                            for i in range(0, len(self.dialog), 2)
                        ])
                        agent.observe(validate(control_msg))
                        acts[idx] = agent.act(timeout=self.max_resp_time)
                        if 'text' in acts[idx]:
                            self.good_rounds.append(acts[idx])

                    # Bad rounds selection
                    for idx, agent in enumerate(self.agents):
                        control_msg['text'] = BAD_ROUND_MSG
                        control_msg['bad_rounds'] = True
                        control_msg['rounds'] = '</ROUND>'.join([
                            '\n'.join(
                                [self.dialog[i][1], self.dialog[i + 1][1]])
                            for i in range(0, len(self.dialog), 2)
                        ])
                        agent.observe(validate(control_msg))
                        acts[idx] = agent.act(timeout=self.max_resp_time)
                        if 'text' in acts[idx]:
                            self.bad_rounds.append(acts[idx])

                    self.chat_done = True
                    for ag in self.agents:
                        ag.observe(validate(acts[idx]))
                        control_msg['text'] = CHAT_ENDED_MSG
                        ag.observe(validate(control_msg))
                return

            self.dialog.append((idx, acts[idx]['text']))
            if self.turn_idx == 1:
                acts[idx]['text'] = self.model_persona_text + '\n' + \
                    acts[idx]['text']
            print(acts[idx])
            self.model_agent.observe(acts[idx])

        # Model_agent turn
        idx = 1
        act = self.model_agent.act()
        acts.append({'text': act['text']})
        if 'reranked_samples' in act:
            acts[-1]['reranked_samples'] = act['reranked_samples']

        for (sb_0, sb_1) in [
            (' .', '.'),
            (' ,', ','),
            (' ?', '?'),
            (' !', '!'),
            ('i ', 'I ')
        ]:
            acts[idx]['text'] = acts[idx]['text'].replace(sb_0, sb_1)
        acts[idx]['text'].capitalize()
        acts[idx]['id'] = 'PERSON_2'
        acts[idx]['message_id'] = acts[0]['message_id'][:-1] + '0' if \
            acts[0]['message_id'][-1] != '0' else \
            acts[0]['message_id'][:-1] + '1'
        self.dialog.append((idx, acts[idx]['text']))
        if 'reranked_samples' in acts[idx]:
            self.reranked_cands.append((idx, acts[idx]['reranked_samples']))
        agent.observe(acts[idx])

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
        for ag in self.agents:
            if (ag.hit_is_abandoned or ag.hit_is_returned or
                    ag.disconnected or ag.hit_is_expired):
                bad_workers.append(ag.worker_id)
                convo_finished = False
        if (not convo_finished or self.dialog == [] or
                self.gmark_score[0] == -1):
            for ag in self.agents:
                ag.not_approve = True
                ag.persona_generator.push_persona(ag.persona_idx)
                print("\n*** Push persona {} back to stack. ****\n".format(
                    ag.persona_idx
                ))
            convo_finished = False

        data_path = self.opt['data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(
                data_path, '{}_{}_{}_{}_withreasons.pkl'.format(
                    self.model_name,
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type
                )
            )
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_{}_incomplete_withreasons.pkl'.format(
                    self.model_name,
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type
                )
            )
        print(
            self.world_tag,
            ': Data successfully saved at {}.'.format(filename)
        )
        self.personas.append(self.agents[0].model_persona[1])
        pickle.dump({'personas': self.personas,
                     'dialog': self.dialog,
                     'workers': [ag.worker_id for ag in self.agents],
                     'hit_id': [ag.hit_id for ag in self.agents],
                     'assignment_id': [ag.assignment_id for ag in self.agents],
                     'bad_workers': bad_workers,
                     'n_turn': self.n_turn,
                     'gmark_score': self.gmark_score,
                     'good_rounds': self.good_rounds,
                     'bad_rounds': self.bad_rounds,
                     'reranked_cands': self.reranked_cands,
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
                per_subseq = [' '.join(per_parse[i:i + len(per_parse) - tolerance])
                              for i in range(tolerance + 1)]
                for pp in per_subseq:
                    if pp in ['', ' ', '  ', '   ']:
                        per_subseq.remove(pp)
                n_word_match += sum([(paa in text) for paa in per_subseq])
            if n_word_match > 0:
                control_msg['text'] = COPIED_CHARACTER_MSG
                ag.observe(validate(control_msg))
                return True
            else:
                return False

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

    def reset_random(self):
        self.n_turn = np.random.randint(
            self.range_turn[0],
            self.range_turn[1]
        ) + 1

    def check_timeout(self, act):
        if ((act['text'] == '[TIMEOUT]') or
                (act['text'] == '[RETURNED]') or (act['text'] == '[DISCONNECT]')):
            control_msg = {'episode_done': True}
            control_msg['id'] = 'SYSTEM'
            control_msg['text'] = self.get_instruction(
                agent_id=act['id'],
                tag='timeout'
            )
            for ag in self.agents:
                if ag.id != act['id']:
                    ag.observe(validate(control_msg))
            self.chat_done = True
            return True
        else:
            return False

    def shutdown(self):
        global shutdown_agent
        gc.collect()

        def shutdown_agent(mturk_agent):
            mturk_agent.shutdown()

        Parallel(
            n_jobs=len(self.agents),
            backend='threading'
        )(delayed(shutdown_agent)(agent) for agent in self.agents)
