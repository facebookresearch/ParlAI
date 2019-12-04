#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.utils.safety import OffensiveStringMatcher
from joblib import Parallel, delayed
from task_config import task_config as config
from extract_and_save_personas import main as main_extract
from constants import (
    MAX_DOC_LEN,
    WIZARD,
    APPRENTICE,
    ONBOARD_MSG,
    APPRENTICE_START_MSG,
    WIZARD_START_MSG,
    TIMEOUT_MSG,
    EXCEED_MIN_TURNS_MSG,
    UNEXPECTED_DISCONNECTION_MSG,
    CHAT_ENDED_MSG,
    STOPWORDS,
    EVAL_WIZARD_MSG,
    PARTNER_RETRIEVED_PASSAGES_INST_MSG,
    WAITING_MSG,
    PICK_TOPIC_MSG,
    AFTER_PICK_TOPIC_MSG,
    AFTER_PICK_TOPIC_WIZARD_MSG,
    AFTER_PARTNER_PICK_TOPIC_WIZARD_MSG,
)
import numpy as np
import time
import os
import pickle
import random
import copy
from urllib.parse import unquote


def split_tokenize(text):
    """
    Splits tokens based on whitespace after adding whitespace around punctuation.
    """
    return (
        text.replace('.', ' . ')
        .replace('. . .', '...')
        .replace(',', ' , ')
        .replace(';', ' ; ')
        .replace(':', ' : ')
        .replace('!', ' ! ')
        .replace('?', ' ? ')
        .replace('(', ' ( ')
        .replace(')', ' ) ')
        .split()
    )


class PersonasGenerator(object):
    def __init__(self, opt):
        self.personas_idx_stack_path = os.path.join(
            os.getcwd(), './personas_idx_stack.pkl'
        )

        self.personas_path = '{}/data/personas-{}'.format(
            os.getcwd(), opt['persona_type'] + 'Original'
        )
        self.topic_for_personas_path = '{}/personas_with_wiki_links.txt'.format(
            os.getcwd()
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

        self.load_topics_for_personas()

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
            os.path.join(self.personas_path, self.personas_name_list[idx]),
            allow_pickle=True,
        )
        return (idx, data)

    def push_persona(self, idx):
        self.idx_stack.append(idx)

    def save_idx_stack(self):
        with open(self.personas_idx_stack_path, 'wb') as handle:
            pickle.dump(self.idx_stack, handle)

    def load_topics_for_personas(self):
        self.persona_to_topics = {}
        with open(self.topic_for_personas_path) as f:
            text = f.read()
            personas = text.split('\n\n')
            for persona in personas:
                persona = persona.split('\n')
                prev_p = persona[0]
                for i in range(1, len(persona)):
                    p_i = persona[i]
                    if 'https' in p_i:
                        topic = unquote(p_i[p_i.rfind('/') + 1 :]).replace('_', ' ')
                        if prev_p in self.persona_to_topics:
                            self.persona_to_topics[prev_p].append(topic)
                        else:
                            self.persona_to_topics[prev_p] = [topic]
                    else:
                        prev_p = p_i

    def get_topics(self, persona_sent):
        return self.persona_to_topics.get(persona_sent, [])


class RoleOnboardWorld(MTurkOnboardWorld):
    """
    A world that provides the appropriate instructions during onboarding.
    """

    def __init__(self, opt, mturk_agent, role):
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_onboard_time = opt['max_onboard_time']
        self.role = role
        super().__init__(opt, mturk_agent)

    def parley(self):
        onboard_msg = {'id': 'SYSTEM'}

        onboard_msg['show_persona'] = False
        onboard_msg['text'] = ONBOARD_MSG
        if self.role == WIZARD:
            onboard_msg['role_task_description'] = config['wizard_onboarding']
        else:
            onboard_msg['role_task_description'] = config['apprentice_onboarding']
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


class MTurkWizardOfWikipediaWorld(MultiAgentDialogWorld):
    """
    World where two agents have a dialogue; one chats freely, perhaps based on a
    persona, while the other is the 'wizard', who bases his/her responses on documents
    (i.e. sentences) retrieved based on what the other agent says.
    """

    def __init__(
        self,
        opt,
        agents=None,
        shared=None,
        world_tag='NONE',
        ir_agent=None,
        task='',
        wiki_title_to_passage=None,
    ):
        self.turn_idx = 0
        self.min_turns = opt['min_turns']
        self.max_turns = opt['max_turns']
        self.num_turns = np.random.randint(self.min_turns, self.max_turns) + 1
        self.dialog = []
        self.wizard_eval = 0
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.chat_done = False
        self.world_tag = world_tag
        self.max_resp_time = opt['max_resp_time']  # in secs
        self.num_passages_to_retrieve = opt['num_passages_retrieved']
        super().__init__(opt, agents, shared)
        self.agents = sorted(agents, key=lambda x: x.id, reverse=random.random() <= 0.5)
        #  Personas and retriever
        self.persona_generator = self.agents[0].persona_generator
        self.relevant_topics = []
        while not self.relevant_topics:
            self.persona_to_topics = {}
            self.persona_idx, persona_data = self.persona_generator.pop_persona()
            for p in persona_data:
                if p[0] == ' ':
                    p = p[1:]
                if p not in self.persona_to_topics:
                    self.persona_to_topics[p] = []
                    topics = set(self.persona_generator.get_topics(p))
                    for t in topics:
                        self.relevant_topics.append(t + ' ({})'.format(p))
                        self.persona_to_topics[p].append(t)

        self.ir_agent = ir_agent
        self.setup_tokenizer(opt)
        self.chosen_topic = ''
        self.chosen_topic_passage = {}
        self.OLD = OffensiveStringMatcher()
        # Load the title to passage dictionary
        self.wiki_title_to_passage = wiki_title_to_passage

    def episode_done(self):
        return self.chat_done

    def setup_tokenizer(self, opt):
        try:
            import nltk
        except ImportError:
            raise ImportError('Please install nltk (e.g. pip install nltk).')
        # nltk-specific setup
        st_path = 'tokenizers/punkt/{0}.pickle'.format(opt['dict_language'])
        try:
            self.sent_tok = nltk.data.load(st_path)
        except LookupError:
            nltk.download('punkt')
            self.sent_tok = nltk.data.load(st_path)

    def sufficient_overlap(self, text, sent_dict):
        text_list = [w[:4] for w in split_tokenize(text.lower()) if w not in STOPWORDS]
        for _, sentence in sent_dict.items():
            sentence_list = [
                w[:4] for w in split_tokenize(sentence.lower()) if w not in STOPWORDS
            ]
            if len(set(text_list).intersection(set(sentence_list))) >= self.opt.get(
                'word_overlap_threshold', 2
            ):
                return True
        return False

    def parley(self):
        """
        Each agent acts; when the APPRENTICE says something, the WIZARD is given
        retrieved documents based on the text response.
        """
        self.turn_idx += 1

        # Initial Message Value
        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))

        '''First Turn: We give the first agent the list of topics to choose from
        '''
        if self.turn_idx == 1:
            for idx, agent in enumerate(self.agents):
                """
                If we are giving the persona, do that :)
                """
                control_msg['text'] = self.get_instruction(
                    tag='start', agent_id=agent.id
                )
                if agent.id == WIZARD:
                    control_msg['description'] = config['wizard_onboarding']
                else:
                    control_msg['description'] = config['apprentice_onboarding']
                agent.observe(validate(control_msg))
                if idx == 0:
                    time.sleep(3)

            '''Send First Person the list of relevant topics'''
            self.agents[0].observe(
                validate(
                    {
                        'id': 'SYSTEM',
                        'text': PICK_TOPIC_MSG,
                        'relevant_topics': self.relevant_topics,
                    }
                )
            )

            topic_act = self.agents[0].act(timeout=self.max_resp_time)
            timed_out = self.check_timeout(topic_act)
            if not timed_out:
                if self.agents[0].id == APPRENTICE:
                    pick_msg = AFTER_PICK_TOPIC_MSG
                else:
                    pick_msg = AFTER_PICK_TOPIC_WIZARD_MSG
                self.agents[0].observe({'id': 'SYSTEM', 'text': pick_msg})
            self.chosen_topic = topic_act['text']

            '''Now, send the wiki page for the chosen topic to the wizard'''
            for idx, agent in enumerate(self.agents):
                if agent.id == WIZARD:
                    passage = self.wiki_title_to_passage.get(self.chosen_topic, '')
                    if passage == '':
                        break
                    split = passage.split('\n')
                    title = split[0]
                    split = self.sent_tok.tokenize(" ".join(split[1:]))
                    split[0] = split[0][1:]
                    sentences = []
                    for sent in split:
                        if len(sent) > 1:
                            sentences.append(sent)
                            if len(" ".join(sentences)) > MAX_DOC_LEN * 2:
                                break
                    msg_text = AFTER_PARTNER_PICK_TOPIC_WIZARD_MSG if idx == 1 else ""
                    control_msg['text'] = msg_text
                    control_msg['chosen_topic_passages'] = [[title, sentences]]
                    agent.observe(validate(control_msg))
                    self.chosen_topic_passage = {
                        'topic': self.chosen_topic,
                        'full_passage': passage,
                        'shown_passage': sentences,
                    }

        '''If we get to the min turns, inform turker that they can end if they
           want
        '''
        if self.turn_idx == self.num_turns + 1:
            for agent in self.agents:
                control_msg['text'] = self.get_instruction(tag='exceed_min_turns')
                control_msg['exceed_min_turns'] = True
                agent.observe(validate(control_msg))

        '''Otherwise, we proceed accordingly'''
        acts = self.acts
        for idx, agent in enumerate(self.agents):
            # Increase response time for wizard
            max_response_time = self.max_resp_time * (
                1 if agent.id == APPRENTICE else 1.5
            )
            acts[idx] = agent.act(timeout=max_response_time)
            self.check_timeout(acts[idx])

            # If chat ends
            if acts[idx]['episode_done']:
                self.chat_done = True
                for ag in self.agents:
                    if ag != agent and ag.some_agent_disconnected:
                        control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                        ag.observe(validate(control_msg))
                        return
                if self.turn_idx > self.num_turns:
                    for ag in self.agents:
                        ag.observe(validate(acts[idx]))
                        '''Have Apprentice Agent Eval Wizard Agent'''
                        if ag.id == APPRENTICE:
                            control_msg['text'] = EVAL_WIZARD_MSG
                            control_msg['wizard_eval'] = True
                            ag.observe(validate(control_msg))
                            act = ag.act(timeout=self.max_resp_time)
                            self.check_timeout(act)
                            try:
                                w_ev = int(act['text'])
                                w_ev = max(w_ev, 1) if w_ev <= 0 else min(w_ev, 5)
                            except ValueError:  # If there is a disconnect here
                                w_ev = -1
                            self.wizard_eval = w_ev
                        control_msg['text'] = CHAT_ENDED_MSG
                        ag.observe(validate(control_msg))
                return

            '''Set up msg info dict to save in dialog'''
            msg_info = {
                'speaker': '{}_{}'.format(idx, agent.id),
                'text': acts[idx]['text'],
                'turn': self.turn_idx,
                'time': time.time(),
                'offensive': self.OLD.contains_offensive_language(acts[idx]['text']),
            }

            '''Get clicked passages and checked sentences from Wizard'''
            if 'clicked_passages' in acts[idx]:
                msg_info['clicked_passages'] = acts[idx]['clicked_passages']
                checked_sents = {}
                for k, v in acts[idx]['checked_sentences'].items():
                    if k == 'no_passages_used':
                        checked_sents[k] = v
                    else:
                        split = k.split('_')
                        person = split[0]
                        topic_idx = split[1]
                        sent_idx = split[2]
                        if person == 'partner':
                            sub_passages = [
                                p.split('\n')[0]
                                for p in self.dialog[-1]['full_passages']
                            ]
                            topic = sub_passages[int(topic_idx)]
                        elif person == 'self':
                            sub_passages = [
                                p.split('\n')[0]
                                for p in self.dialog[-2]['full_passages']
                            ]
                            topic = sub_passages[int(topic_idx)]
                        else:
                            topic = self.chosen_topic
                        cs_key = '_'.join(
                            [person, '_'.join(topic.split(' ')), sent_idx]
                        )
                        checked_sents[cs_key] = v
                msg_info['checked_sentence'] = checked_sents
                msg_info['checked_passage'] = acts[idx]['checked_passages']
                msg_info['good_message'] = self.sufficient_overlap(
                    msg_info['text'], msg_info['checked_sentence']
                )

            '''Retrieve Passages'''
            ir_passages = self.retrieve_passages(copy.deepcopy(acts[idx]))
            passages = self.format_passages(ir_passages)
            msg_info['full_passages'] = ir_passages
            msg_info['shown_passages'] = passages

            if agent.id == WIZARD:
                """
                Give Wizard the Relevant Passages.
                """
                control_msg['text'] = ''
                control_msg['self_retrieved_passages'] = passages
                agent.observe(validate(control_msg))

            self.dialog.append(msg_info)

            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[idx]))
                    if other_agent.id == WIZARD:
                        control_msg['text'] = PARTNER_RETRIEVED_PASSAGES_INST_MSG
                        control_msg['partner_retrieved_passages'] = passages
                        other_agent.observe(validate(control_msg))

    def format_passages(self, ir_passages, max_length=MAX_DOC_LEN):
        passages = []
        if len(ir_passages) == 1:  # Didn't receive any passages
            passages.append(['No Passages Retrieved', []])
        else:
            for passage in ir_passages:
                split = passage.split('\n')
                title = split[0]
                split = self.sent_tok.tokenize(" ".join(split[1:]))
                split[0] = split[0][1:]
                sentences = []
                for sent in split:
                    if len(sent) > 1:
                        sentences.append(sent)
                        if len(" ".join(sentences)) > max_length:
                            break
                passages.append([title, sentences])
        return passages

    def retrieve_passages(self, act, num_passages=None):
        if not num_passages:
            num_passages = self.num_passages_to_retrieve
        self.ir_agent.observe(act)
        action = self.ir_agent.act()
        passages = action.get('text_candidates', [action.get('text', "")])
        return passages[: min(len(passages), num_passages)]

    def get_instruction(self, agent_id=None, tag='first'):
        if tag == 'start':
            start_msg = WIZARD_START_MSG if agent_id == WIZARD else APPRENTICE_START_MSG
            return start_msg.format(self.num_turns)
        if tag == 'timeout':
            return TIMEOUT_MSG
        if tag == 'exceed_min_turns':
            return EXCEED_MIN_TURNS_MSG.format(self.num_turns)

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
        else:
            return False

    def reset_random(self):
        self.num_turns = np.random.randint(self.min_turns, self.max_turns) + 1

    def save_data(self):
        # save persona_idx_stack
        convo_finished = self.turn_idx >= self.num_turns + 1
        for ag in self.agents:
            if (
                ag.hit_is_abandoned
                or ag.hit_is_returned
                or ag.disconnected
                or ag.hit_is_expired
            ):
                convo_finished = False
        if not convo_finished:
            self.persona_generator.push_persona(self.persona_idx)
            print("\n**Push persona {} back to stack. **\n".format(self.persona_idx))
        self.agents[0].persona_generator.save_idx_stack()

        data_path = self.opt['data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.convo_finished = convo_finished
        self.wizard_worker = ''
        if convo_finished:
            filename = os.path.join(
                data_path,
                '{}_{}_{}.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type,
                ),
            )
            self.good_wiz, self.wizard_worker = self.check_wizard_quality()
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_incomplete.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type,
                ),
            )
            self.good_wiz = True
        pickle.dump(
            {
                'persona': self.persona_to_topics,
                'relevant_topics': self.relevant_topics,
                'chosen_topic_passage': self.chosen_topic_passage,
                'dialog': self.dialog,
                'speaker_with_persona': self.agents[0].worker_id,
                'workers': [ag.worker_id for ag in self.agents],
                'n_turn': self.num_turns,
                'hit_ids': [ag.hit_id for ag in self.agents],
                'assignment_ids': [ag.assignment_id for ag in self.agents],
                'wizard_eval': self.wizard_eval,
                'chosen_topic': self.chosen_topic,
                'wizard_good': convo_finished and self.good_wiz,
                'good_wizard_worker': self.wizard_worker if self.good_wiz else '',
                'bad_wizard_worker': self.wizard_worker if not self.good_wiz else '',
            },
            open(filename, 'wb'),
        )
        print('{}: Data successfully saved at {}.'.format(self.world_tag, filename))

    def check_wizard_quality(self):
        """
        Determines whether to soft-block this turker or not Only called if the
        conversation finishes Returns True if the Wizard is good.
        """
        num_good_sents = len(
            list(
                filter(
                    lambda info: 'good_message' in info and info['good_message'],
                    self.dialog,
                )
            )
        )
        wizard_worker = [w for w in self.agents if w.id == WIZARD][0].worker_id
        data_path = self.opt['current_working_dir']
        bad_wizards = os.path.join(data_path, 'bad_wizards.txt')
        good_wizards = os.path.join(data_path, 'good_wizards.txt')
        if num_good_sents < self.opt['num_good_sentence_threshold']:
            if not self.opt['is_sandbox']:
                with open(bad_wizards, 'a') as f:
                    f.write(wizard_worker + '\n')
            return False, wizard_worker
        else:
            if not self.opt['is_sandbox']:
                with open(good_wizards, 'a') as f:
                    f.write(wizard_worker + '\n')
            return True, wizard_worker

    def review_work(self):
        global review_agent

        def review_agent(ag):
            role = ag.id
            for d in self.dialog:
                if role in d['speaker']:
                    if d['offensive']:
                        ag.reject_work(
                            reason='Your HIT has been rejected '
                            'because we detected offensive '
                            'language in your submission.'
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
