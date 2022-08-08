#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
import time
from typing import List, Union, Optional
import unittest

from parlai.core.agents import create_agent
from parlai.core.message import Message

import projects.bb3.prompts as PROMPT
from projects.bb3.agents.module import Module
from projects.bb3.agents.utils import normal_tokenizer, no_overlap, MemoryUtils
from projects.bb3.tests.opt_presets import INIT_OPT


def _self_memory(text):
    return f'{PROMPT.SELF_MEMORY_PREFIX}: {text}'


def _other_memory(text):
    return f'{PROMPT.PARTNER_MEMORY_PREFIX}: {text}'


class TestTokenizer(unittest.TestCase):
    def test_empty(self):
        for txt in ('', '   ', '\n', '\t'):
            tknzd = normal_tokenizer(txt)
            self.assertIsInstance(tknzd, list)
            self.assertEqual(len(tknzd), 0)

    def test_contraction(self):
        txt = "I'd like to see it"
        tknzd = normal_tokenizer(txt)
        self.assertEqual(set(tknzd), {'i', 'would', 'like', 'to', 'see', 'it'})


class TestOverlap(unittest.TestCase):
    # Testing NO overlaps
    def test_no_ovelap_1(self):
        data_list = [_self_memory('I have a banana.'), _self_memory('My car is nice.')]
        self.assertTrue(no_overlap(data_list, _self_memory('I have an orange')))

    def test_no_ovelap_2(self):
        data_list = [_self_memory('I have a banana.'), _self_memory('My car is nice.')]
        self.assertTrue(no_overlap(data_list, _self_memory('I love banana')))

    def test_no_ovelap_3(self):
        data_list = [
            _self_memory('I am traveling to Japan.'),
            _self_memory('My car is nice.'),
        ]
        self.assertTrue(no_overlap(data_list, _other_memory('My car is nice.')))

    def test_no_ovelap_4(self):
        data_list = [
            _self_memory('I am traveling to Japan.'),
            _self_memory('My car is nice.'),
            _self_memory('I have money'),
        ]
        self.assertTrue(no_overlap(data_list, _self_memory('I need money.')))

    def test_no_ovelap_5(self):
        data_list = [
            _self_memory('I am traveling to Japan.'),
            _self_memory('My car is nice.'),
            _self_memory('I have money'),
        ]
        self.assertTrue(no_overlap(data_list, _other_memory('I don\'t have money.')))

    def test_no_ovelap_6(self):
        data_list = [
            _self_memory('I have a banana'),
            _self_memory('I am traveling to Japan.'),
        ]
        self.assertTrue(no_overlap(data_list, _self_memory('I travel to Canada.')))

    # Testig overlaps
    def test_ovelap_1(self):
        data_list = [_self_memory('I have a banana.'), _self_memory('My car is nice.')]
        self.assertFalse(no_overlap(data_list, _self_memory('I have bananas.')))

    def test_ovelap_2(self):
        data_list = [_self_memory('I have a banana'), _self_memory('My car is nice.')]
        self.assertFalse(no_overlap(data_list, _self_memory('My car is nice')))

    def test_ovelap_3(self):
        data_list = [
            _self_memory('I have a banana'),
            _self_memory('My car is really nice.'),
        ]
        self.assertFalse(no_overlap(data_list, _self_memory('My car is nice.')))

    def test_ovelap_4(self):
        data_list = [
            _self_memory('I have a banana'),
            _self_memory('My car is REALLY nice.'),
        ]
        self.assertFalse(no_overlap(data_list, _self_memory('My car is nice.')))

    def test_ovelap_5(self):
        data_list = [
            _self_memory('I have a banana'),
            _self_memory('I am traveling to Japan.'),
        ]
        self.assertFalse(no_overlap(data_list, _self_memory('I travel to Japan.')))


class TestOptFtBase(unittest.TestCase):
    def setUp(self):
        self.opt = INIT_OPT
        for k, v in self.opt.items():
            if (
                v
                == 'parlai_internal.projects.blenderbot3.agents.opt_prompt_agent:PromptAgent'
            ):
                self.opt[
                    k
                ] = 'parlai_internal.projects.blenderbot3.agents.opt_prompt_agent:MockPromptAgent'

        self.opt['search_server'] = 'test'
        self.opt['loglevel'] = 'debug'
        self.opt['override'] = self.opt

        self.message = Message({'text': 'Hey, how is it going?', 'episode_done': False})

        # Opening
        self.memories = [
            MemoryUtils.add_memory_prefix("I am a chatbot", "partner", 'OPT')
        ]
        for animal in ['cats', 'dogs', 'horses', 'parrots']:
            action = random.choice(['like', 'hate', 'have'])
            self.memories.append(
                MemoryUtils.add_memory_prefix(f"I {action} {animal}", 'partner', 'OPT')
            )
        self.opening_message = Message(
            {
                'text': PROMPT.OPENING_PREFIX,
                'episode_done': False,
                'memories': self.memories,
            }
        )


def assert_all_keys_in_replies(
    replies: Union[Message, List[Message]], dialogue_module: Optional[Module]
):
    """
    Assert all keys are in reply.

    For given dialogue module, assert that that specific message is not None.
    """
    if not isinstance(replies, list):
        replies = [replies]
    for reply in replies:
        assert reply['text']
        for basic_key in ['text', 'id', 'memories']:
            assert basic_key in reply

        for m in Module:
            if m is Module.MEMORY_GENERATOR:
                assert all(
                    f'{m.message_name()}_{person}' in reply
                    for person in ['self', 'partner']
                )
            else:
                assert m.message_name() in reply
            if m.is_dialogue():
                assert f"{m.message_name()}_score" in reply
            if not m.skip_search():
                assert all(
                    f'{m.message_name()}_{key}' in reply
                    for key in ['doc_titles', 'doc_content', 'doc_urls']
                )
            if m is dialogue_module:
                assert reply[m.message_name()]


def assert_all_keys_in_obs(observations: Union[Message, List[Message]]):
    if not isinstance(observations, list):
        observations = [observations]
    for observation in observations:
        for m in Module:
            obs = observation[m]
            if 'prompt' not in obs:
                continue
            assert obs['prompt']
            assert obs['prompt'].strip().endswith(f"{m.opt_final_prefix()}:")
            turns = obs['prompt'].split('\n')
            assert all(
                t.strip() not in [PROMPT.SELF_PREFIX, PROMPT.PARTNER_PREFIX]
                for t in turns
            )


class TestOptDecisionCombos(TestOptFtBase):
    # Testing NO overlaps
    def test_decision_combos(self):
        message = self.message
        decisions = ['always', 'never', 'compute']
        for conditioning in ['separate', 'combined']:
            for search_decision in decisions:
                for memory_decision in decisions:
                    for contextual_decision in decisions:
                        opt = copy.deepcopy(self.opt)
                        opt['knowledge_conditioning'] = conditioning
                        opt['override']['knowledge_conditioning'] = conditioning
                        opt['search_decision'] = search_decision
                        opt['override']['search_decision'] = search_decision
                        opt['memory_decision'] = memory_decision
                        opt['override']['memory_decision'] = memory_decision
                        opt['contextual_knowledge_decision'] = contextual_decision
                        opt['override'][
                            'contextual_knowledge_decision'
                        ] = contextual_decision
                        agent = create_agent(opt)
                        agent.observe(message)
                        act = agent.act()
                        if search_decision in ['always', 'compute']:
                            assert Module.SEARCH_DIALOGUE.message_name() in act
                            if conditioning == 'separate':
                                assert act[Module.SEARCH_DIALOGUE.message_name()]
                            assert (
                                Module.SEARCH_KNOWLEDGE.message_name() in act
                                and act[Module.SEARCH_KNOWLEDGE.message_name()]
                            )
                        if memory_decision in ['always', 'compute']:
                            # TODO: Something is wrong when memory_decision is always,
                            # search decision never, ctxt never
                            agent.observe(message)
                            act2 = agent.act()
                            assert Module.MEMORY_DIALOGUE.message_name() in act2
                            always_only_memory = (
                                search_decision == 'never'
                                and contextual_decision == 'never'
                            )
                            if always_only_memory:
                                # sometimes this doesn't work
                                assert (
                                    act2[Module.MEMORY_DIALOGUE.message_name()]
                                    or act2[Module.VANILLA_DIALOGUE.message_name()]
                                )
                            else:
                                if conditioning == 'separate':
                                    assert act2[Module.MEMORY_DIALOGUE.message_name()]
                                assert (
                                    Module.MEMORY_KNOWLEDGE.message_name() in act2
                                    and act2[Module.MEMORY_KNOWLEDGE.message_name()]
                                )
                        if contextual_decision == 'always' or (
                            contextual_decision == 'compute'
                            and search_decision == 'never'
                            and memory_decision == 'never'
                        ):
                            assert Module.CONTEXTUAL_DIALOGUE.message_name() in act
                            if conditioning == 'separate':
                                act[Module.CONTEXTUAL_DIALOGUE.message_name()]
                            assert (
                                Module.CONTEXTUAL_KNOWLEDGE.message_name() in act
                                and act[Module.CONTEXTUAL_KNOWLEDGE.message_name()]
                            )


class TestOptOpening(TestOptFtBase):
    def test_opening(self):
        opt = copy.deepcopy(self.opt)
        opt['knowledge_conditioning'] = 'separate'
        opt['override']['knowledge_conditioning'] = 'separate'
        agent = create_agent(opt)

        # Check that we get an opener when formatted correctly
        agent.observe(self.opening_message)
        prompt = agent.observations[Module.OPENING_DIALOGUE].get('prompt')
        assert prompt
        assert prompt == '\n'.join(self.memories + [f"{PROMPT.OPENING_PREFIX}:"])
        act = agent.act()
        assert_all_keys_in_replies(act, Module.OPENING_DIALOGUE)

        # check we do not get an opener on the second message, despite having memories
        message2 = copy.deepcopy(self.message)
        message2.force_set('memories', self.memories)
        agent.observe(message2)
        act = agent.act()
        # memories now has one more entry
        assert len(agent.memories) == len(self.memories)
        assert_all_keys_in_replies(act, Module.SEARCH_DIALOGUE)
        assert_all_keys_in_replies(act, Module.MEMORY_DIALOGUE)

        # check that a third message still does not duplicate memories
        message3 = copy.deepcopy(self.message)
        agent.observe(message3)
        agent.act()
        assert len(agent.memories) == len(self.memories)

        # check we do not get an opener if we send OPENING_PREFIX, but no memories
        agent.reset()
        message3 = copy.deepcopy(self.opening_message)
        message3.force_set('text', PROMPT.OPENING_PREFIX)
        message3.pop('memories', None)
        agent.observe(message3)
        act = agent.act()
        assert not act[Module.OPENING_DIALOGUE.message_name()]

    def test_failed_opening(self):
        opt = copy.deepcopy(self.opt)
        opt['knowledge_conditioning'] = 'separate'
        opt['override']['knowledge_conditioning'] = 'separate'
        agent = create_agent(opt)
        agent.agents[
            Module.OPENING_DIALOGUE
        ].mock_text = f"{PROMPT.PARTNER_MEMORY_PREFIX}: I am a chatter"
        agent.batch_agents[
            Module.OPENING_DIALOGUE
        ].mock_text = f"{PROMPT.PARTNER_MEMORY_PREFIX}: I am a chatter"
        agent.observe(self.opening_message)
        act = agent.act()
        assert PROMPT.PARTNER_MEMORY_PREFIX not in act['text']


class TestOptBatching(TestOptFtBase):
    def test_batching(self):
        opt = copy.deepcopy(self.opt)
        opt['memory_decision'] = 'always'
        opt['override']['memory_decision'] = 'always'
        opt['knowledge_conditioning'] = 'combined'
        opt['override']['knowledge_conditioning'] = 'combined'

        agent1 = create_agent(opt)
        agent2 = agent1.clone()
        obs1, obs2 = agent1.observe(self.message), agent2.observe(self.message)
        for agent, reply in zip([agent1, agent2], agent1.batch_act([obs1, obs2])):
            agent.self_observe(reply)
        obs1, obs2 = agent1.observe(self.message), agent2.observe(self.message)
        agent1.batch_act([obs1, obs2])


class TestOptMainServerBase(TestOptFtBase):
    def setUp(self):
        super().setUp()
        opt = copy.deepcopy(self.opt)
        overrides = {
            'knowledge_conditioning': 'combined',
            'opt_server': 'http://18.117.126.138:3000',
            'search_server': 'bing_cc',
        }
        for k, v in overrides.items():
            opt[k] = v
            opt['override'][k] = v
        self.opt = opt

        self.batch_agent = create_agent(opt)


class TestOptMainServerBatching(TestOptMainServerBase):
    def test_batching(self):
        self.batch_agent = create_agent(self.opt)
        agents = [self.batch_agent.clone() for _ in range(8)]
        observations = [a.observe(self.opening_message) for a in agents]
        assert_all_keys_in_obs(observations)
        replies = self.batch_agent.batch_act(observations)
        assert_all_keys_in_replies(replies, Module.OPENING_DIALOGUE)
        text_choices = [
            "That's so cool! What else do you like to do for fun?",
            "How many people live there?",
            "How much money does something like that cost?",
            "Wow, can you maybe tell me a joke?",
            "Do you know anything cool about the new york yankees?",
            "Do you know anything cool about the new york mets?",
            "Do you know anything cool about astronomy",
            "Neat!",
            "Who is the oldest living person?",
            "Do you have any other pets?",
        ]
        # more rounds
        for _ in range(10):
            time.sleep(2)
            # 10 rounds.
            observations = []
            for agent in agents:
                message = copy.deepcopy(self.message)
                message.force_set('text', random.choice(text_choices))
                observations.append(agent.observe(message))

            assert_all_keys_in_obs(observations)
            replies = self.batch_agent.batch_act(observations)
            assert_all_keys_in_replies(replies, None)


class TestInjectQueryString(TestOptFtBase):
    def test_inject_query_string(self):
        opt = copy.deepcopy(self.opt)
        inject_string = 'I AM AN INJECTION'
        opt['inject_query_string'] = inject_string
        opt['override']['inject_query_string'] = inject_string
        opt['search_decision'] = 'always'
        opt['override']['search_decision'] = 'always'

        agent = create_agent(opt)
        agent.observe(self.message)
        act = agent.act()
        assert act[Module.SEARCH_QUERY.message_name()]
        assert act[Module.SEARCH_QUERY.message_name()].endswith(inject_string)


class TestMemoryTFIDF(TestOptFtBase):
    def test_memory_tfidf(self):
        agent = create_agent(self.opt)
        dictionary = agent.dictionary
        memories = self.memories * 100
        new_memories = MemoryUtils.get_available_memory(
            {'text': 'I wish I could see my cats again!', 'memories': memories},
            dictionary,
        )
        assert "cats" in new_memories[0]
        assert len(new_memories) <= 32
        new_memories = MemoryUtils.get_available_memory(
            {'text': 'I hope the horses are faster today!', 'memories': memories},
            dictionary,
        )
        assert "horses" in new_memories[0]
        assert len(new_memories) <= 32
