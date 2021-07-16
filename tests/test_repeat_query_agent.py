#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for RepeatQueryAgent.
"""

import unittest
from parlai.core.agents import create_agent
from parlai.core.message import Message


class TestRepeatQueryAgent(unittest.TestCase):
    def test_respond(self):
        """
        Tests respond() where the agent provides a string response to a single message.
        """
        agent = create_agent(dict(model='repeat_query'))
        message = Message(
            {
                'text': 'hi!',
                'label': ['A'],
                'episode_done': False,
                'label_candidates': ['A', 'B', 'C'],
            }
        )
        response = agent.respond(message)
        self.assertEqual(response, 'hi!')
        message = Message({'text': 'hello!', 'episode_done': False})
        response = agent.respond(message, label=['A'])
        self.assertEqual(response, 'hello!')
        response = agent.respond(Message(text='no way!'), label=['A'])
        self.assertEqual(response, 'no way!')
        response = agent.respond('what\'s up?', episode_done=True)
        self.assertEqual(response, 'what\'s up?')
        response = agent.respond('hey there!')
        self.assertEqual(response, 'hey there!')
        response = agent.respond('')
        self.assertEqual(response, 'Nothing to repeat yet.')
        response = agent.respond(Message(episode_done=True), text='I feel infinite.')
        self.assertEqual(response, 'I feel infinite.')

    def test_respond_error(self):
        """
        Tests respond() when it errors out.
        """
        agent = create_agent(dict(model='repeat_query'))
        error_message = 'The agent needs a \'text\' field in the message.'
        with self.assertRaises(Exception) as context:
            agent.respond(Message(episode_done=True))
        self.assertEqual(str(context.exception), error_message)
        with self.assertRaises(Exception) as context:
            agent.respond({})
        self.assertEqual(str(context.exception), error_message)
        with self.assertRaises(Exception) as context:
            agent.respond(Message())
        self.assertEqual(str(context.exception), error_message)

    def test_batch_respond(self):
        """
        Tests batch_respond() of Repeat Query agent.
        """
        agent = create_agent(dict(model='repeat_query'))
        messages = [
            Message({'text': 'hello!', 'episode_done': False}),
            Message({'text': 'hi!', 'episode_done': False}),
            Message({'text': 'what\'s up?', 'episode_done': False}),
            Message({'text': '', 'episode_done': False}),
            Message({'text': 'I feel infinite.', 'episode_done': False}),
        ]
        expected_response = [
            'hello!',
            'hi!',
            'what\'s up?',
            'Nothing to repeat yet.',
            'I feel infinite.',
        ]
        batch_response = agent.batch_respond(messages)
        self.assertEqual(batch_response, expected_response)

    def test_batch_act(self):
        """
        Tests batch_act() of Repeat Query agent.
        """
        agent = create_agent(dict(model='repeat_query'))
        observations = []
        batch_reply = agent.batch_act(observations)
        self.assertEqual(len(batch_reply), 0)
        observations = [
            Message({'text': 'hello!', 'episode_done': False}),
            Message({'text': '', 'episode_done': False}),
            Message({'episode_done': False}),
            Message(),
            None,
        ]
        original_obs = "Hey there!"
        agent.observe(original_obs)
        self.assertEqual(agent.observation, original_obs)
        batch_reply = agent.batch_act(observations)
        # Make sure original observation doesn't change.
        self.assertEqual(agent.observation, original_obs)
        self.assertEqual(len(batch_reply[0]), 3)
        self.assertEqual(batch_reply[0]['text'], 'hello!')
        self.assertEqual(batch_reply[0]['episode_done'], False)
        self.assertEqual(batch_reply[0]['id'], 'RepeatQueryAgent')
        self.assertEqual(len(batch_reply[1]), 3)
        self.assertEqual(batch_reply[1]['text'], 'Nothing to repeat yet.')
        self.assertEqual(batch_reply[1]['episode_done'], False)
        self.assertEqual(batch_reply[1]['id'], 'RepeatQueryAgent')
        self.assertEqual(len(batch_reply[2]), 3)
        self.assertEqual(batch_reply[2]['text'], "I don't know")
        self.assertEqual(batch_reply[2]['episode_done'], False)
        self.assertEqual(batch_reply[2]['id'], 'RepeatQueryAgent')
        self.assertEqual(len(batch_reply[3]), 3)
        self.assertEqual(batch_reply[3]['text'], "I don't know")
        self.assertEqual(batch_reply[3]['episode_done'], False)
        self.assertEqual(batch_reply[3]['id'], 'RepeatQueryAgent')
        self.assertEqual(len(batch_reply[4]), 2)
        self.assertEqual(batch_reply[4]['text'], 'Nothing to repeat yet.')
        self.assertEqual(batch_reply[4]['episode_done'], False)
