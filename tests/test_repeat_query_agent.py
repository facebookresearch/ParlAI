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
