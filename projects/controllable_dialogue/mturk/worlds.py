#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.legacy_2018.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.legacy_2018.worlds import MTurkOnboardWorld
from parlai.core.message import Message
from parlai.utils.strings import normalize_reply

from joblib import Parallel, delayed
import numpy as np
import os
import json
import random
import time
import torch
import copy


# ASK_DETAILED decides whether we ask human evaluators to select individual
# utterances they found bad. The See et al. 2019 paper has this as True; it is
# set False in later works as it adds overhead that isn't used in analysis.
ASK_DETAILED = False

# INSTRUCTIONS
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
        the character descriptions into the message.</b></span> \n \
        <span style="color:red"><b>If you see this message twice, please \
        return the hit and accept the next one.</b></span>'
CHAT_NOT_DONE_MSG = 'Sorry, we need at least <b>{} more turn(s)</b> to finish. \
       Please send a new message:'
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
EXCEED_MIN_TURNS_MSG = '\n {} chat turns finished! \n \
        You can click the "Done" button to end the chat if it\'s your turn '
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
# Engagingness
ENGAGINGNESS_MSGS = [
    'How much did you enjoy talking to this user?',
    # 'How likely would you be to continue talking to this user?',
]
ENGAGINGNESS_CHOICES = ['not at all', 'a little', 'somewhat', 'a lot']

INTERESTINGNESS_MSGS = ['How interesting or boring did you find this conversation?']
INTERESTINGNESS_CHOICES = [
    'Very boring',
    'A little boring',
    'A little interesting',
    'Very interesting',
]

LISTENING_MSGS = ['How much did the user seem to pay attention to what you said?']
LISTENING_CHOICES = [
    'Always ignored what I said',
    'Mostly ignored what I said',
    'Mostly paid attention to what I said',
    'Always paid attention to what I said',
]

INQUISITIVENESS_MSGS = ['How much did the user try to get to know you?']
INQUISITIVENESS_CHOICES = [
    "Didn't ask about me at all",
    "Asked about me some",
    "Asked about me a good amount",
    "Asked about me too much",
]

REPETITIVENESS_MSGS = [
    'How repetitive was this user?',
    'Please select the sentences that you found repetitive:',
]
REPETITIVENESS_CHOICES = [
    'Repeated themselves over and over',
    'Sometimes said the same thing twice',
    'Always said something new',
]

# Fluency
FLUENCY_MSGS = [
    "How naturally did this user speak English?",
    'Please select the sentences containing unnatural English:',
]
FLUENCY_CHOICES = [
    'Very unnatural',
    'Mostly unnatural',
    'Mostly natural',
    'Very natural',
]

# Consistency
CONSISTENCY_MSGS = [
    "How often did this user say something which did <b>NOT</b> make sense?",
    ("Please select the sentences which did <b>NOT</b> make sense:"),
]
CONSISTENCY_CHOICES = [
    'Everything made perfect sense',
    "Some responses didn't make sense",
    "Most responses didn't make sense",
    'Never made any sense',
]

HUMANNESS_MSGS = ['Do you think this user is a bot or a human?']
HUMANNESS_CHOICES = [
    'Definitely a bot',
    'Probably a bot',
    'Probably a human',
    'Definitely a human',
]

# Persona
PERSONA_MSG = (
    'Which prompt (character) do you think the other user was '
    + 'given for this conversation?  \n 1.<br> {} <br> 2.<br> {}'
)
PERSONA_CHOICES = ['1', '2']


def _strip_tensors(act):
    """
    Remove all tensor objects from an act to ensure we don't try to serialize them.
    """
    return Message({k: v for k, v in act.items() if not torch.is_tensor(v)})


def _random_delay():
    time.sleep(max(0, 4 + np.random.randn() * 0.5))


class PersonasGenerator(object):
    def __init__(self, opt):
        self.text_file = self._path(opt)
        self.personas = self.extract_personas()

    def _path(self, opt):
        # Build the data if it doesn't exist.
        persona = opt['persona_type']
        datatype = opt['persona_datatype'].split(':')[0]
        dt = datatype + '_' + persona
        if datatype == 'test':
            return os.path.join(
                opt['parlai_home'],
                'parlai_internal/projects/convai2/test_set',
                dt + '_original_no_cands.txt',
            )
        return os.path.join(opt['datapath'], 'ConvAI2', dt + '_original_no_cands.txt')

    def extract_personas(self):
        personas = []
        with open(self.text_file, 'r') as f:
            lines = f.readlines()

        new_persona = []
        for line in lines:
            if 'persona: ' in line:
                new_persona.append(line.split('persona: ')[1].replace('\n', ''))
            else:
                if new_persona:
                    personas.append(new_persona)
                    new_persona = []

        return personas

    def get_persona(self):
        return random.choice(self.personas)


class PersonaAssignWorld(MTurkOnboardWorld):
    """
    A world that assigns a persona to an agent.
    """

    def __init__(self, opt, mturk_agent):
        self.max_persona_time = opt['max_persona_time']
        self.human_eval = opt['human_eval']
        super().__init__(opt, mturk_agent)

    def parley(self):
        personas = self.mturk_agent.personas_generator.get_persona()
        self.mturk_agent.personas = personas
        if not self.human_eval:
            # get model personas
            model_personas = self.mturk_agent.personas_generator.get_persona()
            while model_personas == personas:
                model_personas = self.mturk_agent.personas_generator.get_persona()
            self.mturk_agent.model_personas = model_personas

        persona_text = ''
        for persona in personas:
            persona_text += '<b><span style="color:blue">' '{}\n</span></b>'.format(
                persona.strip()
            )

        self.mturk_agent.observe(
            {
                'id': 'SYSTEM',
                'show_persona': True,
                'text': ONBOARD_MSG + '<br>' + persona_text + '<br>',
            }
        )

        act = self.mturk_agent.act(timeout=self.max_persona_time)
        timed_out = self.check_timeout(act)
        if timed_out:
            self.episodeDone = True
            return

    def check_timeout(self, act):
        if 'text' in act:
            if (
                (act['text'] == '[TIMEOUT]')
                or (act['text'] == '[RETURNED]')
                or (act['text'] == '[DISCONNECT]')
            ):
                return True
        return False


class ControllableDialogEval(MultiAgentDialogWorld):
    def __init__(
        self,
        opt,
        agents=None,
        shared=None,
        num_turns=6,
        max_resp_time=120,
        model_agent_opt=None,
        world_tag='',
        agent_timeout_shutdown=120,
        model_config=None,
    ):

        # TURN CONTROL
        self.opt = opt
        self.turn_idx = 0
        self.n_turn = num_turns
        self.chat_done = False
        self.other_first = random.choice([True, False])
        self.model_config = model_config

        # DATA
        self.start_time = time.time()
        self.dialog = []
        self.dialog_list = []

        self.engagingness_scores = []
        self.interestingness_scores = []
        self.listening_scores = []
        self.consistency_scores = []
        self.inquisitiveness_scores = []
        self.humanness_scores = []
        self.repetitiveness_scores = []
        self.fluency_scores = []
        self.persona_scores = []

        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.world_tag = world_tag

        super().__init__(opt, agents, shared)

        # MODEL AGENT SET UP
        if model_agent_opt is not None:
            self.model_agent = create_agent_from_shared(model_agent_opt)
        else:
            # case where we test against a human
            self.model_agent = None

        # TIMEOUT PROTOCOLS
        self.max_resp_time = max_resp_time  # in secs
        self.agent_timeout_shutdown = agent_timeout_shutdown

        # PERSONAS
        self.bot_seen_persona = False
        self.personas = [ag.personas for ag in self.agents]
        if self.model_agent is not None:
            self.eval_agent = self.agents[0]
            self.model_personas = self.agents[0].model_personas
            self.model_persona_text = '\n'.join(
                ['your persona: ' + pers for pers in self.model_personas]
            )
        else:
            self.model_personas = None
            for idx in range(len(self.agents)):
                if self.agents[idx].id == 'PERSON_1':
                    self.eval_agent = self.agents[idx]
                    self.other_agent = self.agents[idx - 1]
                    break

    def get_control_msg(self):
        return {'id': 'SYSTEM', 'episode_done': False}

    def get_human_agent_act(self, agent):
        act = agent.act(timeout=self.max_resp_time)
        while self.is_msg_tooshortlong(act, agent):
            act = agent.act(timeout=self.max_resp_time)
        return act

    def format_personachat_text(self, text):
        new_text = text.lower()

        switch_list = [
            ("we're", "were"),
            ("let's", "lets"),
            ("it's", "its"),
            ("who's", "whos"),
            ("you're", "youre"),
            ("you've", "youve"),
            ("he'd", "hed"),
            ("he'll", "hell"),
        ]
        for tup in switch_list:
            new_text = new_text.replace(tup[0], tup[1])

        return new_text

    def get_bot_observation(self):
        # TODO: clear bots queue each time so that it observes itself properly
        pass

    def parley(self):
        self.turn_idx += 1
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
                control_msg = self.get_control_msg()
                control_msg['persona_text'] = persona_text
                control_msg['text'] = self.get_instruction(
                    tag='start', agent_id=agent.id
                )
                # TODO: check that get instruction actually exists?
                agent.observe(validate(control_msg))
                if idx == 0:
                    time.sleep(3)

        """If we get to the min turns, inform turker that they can end if they
        want.
        """
        if self.turn_idx == self.n_turn + 1:
            for idx, agent in enumerate(self.agents):
                control_msg = self.get_control_msg()
                control_msg['text'] = self.get_instruction(idx, tag='exceed_min_turns')
                control_msg['exceed_min_turns'] = True
                agent.observe(validate(control_msg))

        """Otherwise, we proceed accordingly."""
        # Other agent first
        if self.other_first and self.turn_idx == 1:
            if self.model_agent is not None:
                # Model must observe its persona
                persona_act = {
                    'text': '\n'.join([self.model_persona_text, '__SILENCE__']),
                    'episode_done': False,
                }
                self.model_agent.observe(persona_act)
                self.bot_seen_persona = True
                model_act = copy.deepcopy(self.model_agent.act())
                model_act.force_set('text', normalize_reply(model_act['text']))
                model_act.force_set('id', 'PERSON_2')
                self.dialog.append((1, model_act.get('text')))
                _random_delay()
                self.eval_agent.observe(_strip_tensors(model_act))
            else:
                act = self.get_human_agent_act(self.other_agent)
                timeout = self.check_timeout(act)
                if timeout:
                    # eval agent early disconnect
                    control_msg = self.get_control_msg()
                    control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                    self.eval_agent.observe(validate(control_msg))
                    return
                else:
                    self.dialog.append((1, act.get('text')))
                    act = copy.deepcopy(act)
                    act.force_set('text', normalize_reply(act['text']))
                    self.eval_agent.observe(act)

        # Eval agent turn
        act = Message(self.get_human_agent_act(self.eval_agent))
        timeout = self.check_timeout(act)
        if timeout:
            if self.model_agent is None:
                control_msg = self.get_control_msg()
                control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                self.other_agent.observe(validate(control_msg))
            return

        if act['episode_done']:
            if self.turn_idx >= self.n_turn:
                if not self.other_first:
                    self.dialog_list = [
                        '\n'.join([self.dialog[i][1], self.dialog[i + 1][1]])
                        for i in range(0, len(self.dialog), 2)
                    ]
                else:
                    self.dialog_list = [' \n' + self.dialog[0][1]] + [
                        '\n'.join([self.dialog[i][1], self.dialog[i + 1][1]])
                        for i in range(1, len(self.dialog) - 1, 2)
                    ]
                self.parallel_eval_mode()

                self.chat_done = True
                for ag in self.agents:
                    control_msg = self.get_control_msg()
                    control_msg['text'] = CHAT_ENDED_MSG
                    ag.observe(validate(control_msg))
            return

        self.dialog.append((0, act['text']))

        if not self.bot_seen_persona and self.model_agent is not None:
            # Add persona for model to observe
            act.force_set('text', '\n'.join([self.model_persona_text, act['text']]))
            self.bot_seen_persona = True
        if self.model_agent is not None:
            self.model_agent.observe(act)
        else:
            act = copy.deepcopy(act)
            act.force_set('text', normalize_reply(act['text']))
            self.other_agent.observe(act)

        # Model_agent turn
        if not self.other_first or self.turn_idx < self.n_turn:
            if self.model_agent is not None:
                _random_delay()
                act = _strip_tensors(copy.deepcopy(self.model_agent.act()))
                act.force_set('text', normalize_reply(act['text']))
                act.force_set('id', 'PERSON_2')
                # NOTE: your model may or may not need to observe itself here
                # If it does, call model_observes_itself or some other specialized
                # function
            else:
                act = self.get_human_agent_act(self.other_agent)
                timeout = self.check_timeout(act)
                if timeout:
                    # eval agent early disconnect
                    control_msg = self.get_control_msg()
                    control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                    self.eval_agent.observe(validate(control_msg))
                    return

            self.dialog.append((1, act.get('text')))
            act = copy.deepcopy(act)
            act.force_set('text', normalize_reply(act['text']))
            self.eval_agent.observe(act)

    def _evaluate_characteristic(self, question, choices, addto):
        control_msg = self.get_control_msg()
        control_msg['text'] = question
        control_msg['button_choices'] = '</ROUND>'.join(choices)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = choices.index(act.get('text'))
        addto.append(act_choice)
        return True

    def evaluate_engagingness(self):
        control_msg = self.get_control_msg()
        msg_rng = len(ENGAGINGNESS_MSGS)
        for i in range(msg_rng):
            control_msg['text'] = ENGAGINGNESS_MSGS[i]
            control_msg['button_choices'] = '</ROUND>'.join(ENGAGINGNESS_CHOICES)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            act_choice = ENGAGINGNESS_CHOICES.index(act.get('text'))
            self.engagingness_scores.append(act_choice)
        return True

    def evaluate_interestingness(self):
        return self._evaluate_characteristic(
            INTERESTINGNESS_MSGS[0],
            INTERESTINGNESS_CHOICES,
            self.interestingness_scores,
        )

    def evaluate_listening(self):
        return self._evaluate_characteristic(
            LISTENING_MSGS[0], LISTENING_CHOICES, self.listening_scores
        )

    def evaluate_repetitiveness(self):
        control_msg = self.get_control_msg()
        control_msg['text'] = REPETITIVENESS_MSGS[0]
        control_msg['button_choices'] = '</ROUND>'.join(REPETITIVENESS_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = REPETITIVENESS_CHOICES.index(act.get('text'))
        self.repetitiveness_scores.append(act_choice)
        if ASK_DETAILED and act_choice != 2:
            control_msg = self.get_control_msg()
            control_msg['text'] = REPETITIVENESS_MSGS[1]
            control_msg['good_rounds'] = True
            control_msg['rounds'] = '</ROUND>'.join(self.dialog_list)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            if 'text' in act:
                self.repetitiveness_scores.append(
                    [int(x) - 1 for x in act['text'].split(',')]
                )
        return True

    def evaluate_inquisitiveness(self):
        return self._evaluate_characteristic(
            INQUISITIVENESS_MSGS[0],
            INQUISITIVENESS_CHOICES,
            self.inquisitiveness_scores,
        )

    def evaluate_humanness(self):
        return self._evaluate_characteristic(
            HUMANNESS_MSGS[0], HUMANNESS_CHOICES, self.humanness_scores
        )

    def evaluate_fluency(self):
        control_msg = self.get_control_msg()
        control_msg['text'] = FLUENCY_MSGS[0]
        control_msg['button_choices'] = '</ROUND>'.join(FLUENCY_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = FLUENCY_CHOICES.index(act.get('text'))
        self.fluency_scores.append(act_choice)
        if ASK_DETAILED and act_choice != 3:
            control_msg = self.get_control_msg()
            control_msg['text'] = FLUENCY_MSGS[1]
            control_msg['good_rounds'] = True
            control_msg['rounds'] = '</ROUND>'.join(self.dialog_list)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            if 'text' in act:
                self.fluency_scores.append([int(x) - 1 for x in act['text'].split(',')])
        return True

    def evaluate_consistency(self):
        control_msg = self.get_control_msg()
        control_msg['text'] = CONSISTENCY_MSGS[0]
        control_msg['button_choices'] = '</ROUND>'.join(CONSISTENCY_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = CONSISTENCY_CHOICES.index(act.get('text'))
        self.consistency_scores.append(act_choice)
        if ASK_DETAILED and act_choice != 0:
            control_msg = self.get_control_msg()
            control_msg['text'] = CONSISTENCY_MSGS[1]
            control_msg['good_rounds'] = True
            control_msg['rounds'] = '</ROUND>'.join(self.dialog_list)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            if 'text' in act:
                self.consistency_scores.append(
                    [int(x) - 1 for x in act['text'].split(',')]
                )
        return True

    def evaluate_persona(self):
        if self.model_agent is not None:
            other_persona = self.model_personas
        else:
            other_persona = self.other_agent.personas
        fake_persona = self.eval_agent.personas_generator.get_persona()
        while fake_persona == other_persona:
            fake_persona = self.eval_agent.personas_generator.get_persona()

        cand_text = []
        for dt in [other_persona, fake_persona]:
            if dt == other_persona:
                is_correct = True
            else:
                is_correct = False
            _text = ''
            for s in dt:
                _text += '<b><span style="color:blue">' + s.strip() + '</span></b><br>'
            cand_text.append((is_correct, _text))
        random.shuffle(cand_text)

        control_msg = self.get_control_msg()
        control_msg['text'] = PERSONA_MSG.format(cand_text[0][1], cand_text[1][1])
        control_msg['button_choices'] = '</ROUND>'.join(PERSONA_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False

        self.persona_scores.append(cand_text[int(act['text']) - 1][0])
        return True

    def parallel_eval_mode(self):
        """
        Parallel function that shuts one agent down and asks the other to do the
        evaluation if their are two agents.

        If there is only one agent, it performs the evaluation.
        """

        def eval_or_shutdown(agent):
            if self.model_agent is None and agent == self.other_agent:
                control_msg = self.get_control_msg()
                control_msg['text'] = OTHER_AGENT_FINISHED_MSG
                self.other_agent.observe(validate(control_msg))
                # mark eval agent done
                self.eval_agent.mturk_manager.mark_workers_done([self.eval_agent])
                # shutdown other agent
                self.other_agent.shutdown()
            else:
                evaluations = [
                    self.evaluate_engagingness,
                    self.evaluate_interestingness,
                    self.evaluate_inquisitiveness,
                    self.evaluate_listening,
                    self.evaluate_repetitiveness,
                    self.evaluate_fluency,
                    self.evaluate_consistency,
                    self.evaluate_humanness,
                    self.evaluate_persona,
                ]
                for evaluation in evaluations:
                    fin = evaluation()
                    if not fin:
                        return
                return

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
        if self.dialog == [] or self.persona_scores == []:
            convo_finished = False

        self.convo_finished = convo_finished
        data_path = self.opt['save_data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(
                data_path,
                '{}_{}_{}.json'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type,
                ),
            )
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_incomplete.json'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type,
                ),
            )
        json.dump(
            {
                'dialog': self.dialog,
                'dialog_list': self.dialog_list,
                'other_first': self.other_first,
                'bot_went_first': self.other_first,
                'start_time': self.start_time,
                'timestamp': time.time(),
                'total_time': time.time() - self.start_time,
                'workers': [ag.worker_id for ag in self.agents],
                'hit_id': [ag.hit_id for ag in self.agents],
                'assignment_id': [ag.assignment_id for ag in self.agents],
                'human_personas': [ag.personas for ag in self.agents],
                'model_personas': self.model_personas,
                'bad_workers': bad_workers,
                'n_turn': self.n_turn,
                'engagingness': self.engagingness_scores,
                'interestingness': self.interestingness_scores,
                'listening': self.listening_scores,
                'consistency': self.consistency_scores,
                'inquisitiveness': self.inquisitiveness_scores,
                'repetitiveness': self.repetitiveness_scores,
                'humanness': self.humanness_scores,
                'fluency': self.fluency_scores,
                'persona': self.persona_scores,
                'opt': self.opt,
                'model_config': self.model_config,
            },
            open(filename, 'w'),
        )
        print(self.world_tag, ': Data successfully saved at {}.'.format(filename))

    def is_msg_tooshortlong(self, act, ag, th_min=3, th_max=20):
        if act['episode_done']:
            return False

        control_msg = self.get_control_msg()

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
        pass

    def check_timeout(self, act):
        if act is None:
            self.chat_done = True
            return True
        if (
            (act['text'] == '[TIMEOUT]')
            or (act['text'] == '[RETURNED]')
            or (act['text'] == '[DISCONNECT]')
        ):
            control_msg = self.get_control_msg()
            control_msg['episode_done'] = True
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
