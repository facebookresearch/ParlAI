import unittest
from parlai.core.agents import create_agent
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
import parlai.utils.testing as testing_utils


class TestRepeatQueryAgent(unittest.TestCase):
    def test_respond(self):
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
        agent = create_agent(dict(model='repeat_query'))
        error_message = 'The agent needs a \'text\' field in the message.'
        with self.assertRaises(Exception) as context:
            response = agent.respond(Message(episode_done=True))
        self.assertEqual(str(context.exception), error_message)
        with self.assertRaises(Exception) as context:
            response = agent.respond({})
        self.assertEqual(str(context.exception), error_message)
        with self.assertRaises(Exception) as context:
            response = agent.respond(Message())
        self.assertEqual(str(context.exception), error_message)
