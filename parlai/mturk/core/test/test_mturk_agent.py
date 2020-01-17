#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import os
import time
import threading
from unittest import mock
from parlai.mturk.core.agents import MTurkAgent
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser

import parlai.mturk.core.worker_manager as WorkerManagerFile
import parlai.mturk.core.data_model as data_model

parent_dir = os.path.dirname(os.path.abspath(__file__))
WorkerManagerFile.DISCONNECT_FILE_NAME = 'disconnect-test.pickle'
WorkerManagerFile.MAX_DISCONNECTS = 1
WorkerManagerFile.parent_dir = os.path.dirname(os.path.abspath(__file__))

TEST_WORKER_ID_1 = 'TEST_WORKER_ID_1'
TEST_ASSIGNMENT_ID_1 = 'TEST_ASSIGNMENT_ID_1'
TEST_HIT_ID_1 = 'TEST_HIT_ID_1'
TEST_CONV_ID_1 = 'TEST_CONV_ID_1'
FAKE_ID = 'BOGUS'

MESSAGE_ID_1 = 'MESSAGE_ID_1'
MESSAGE_ID_2 = 'MESSAGE_ID_2'
COMMAND_ID_1 = 'COMMAND_ID_1'

MESSAGE_TYPE = data_model.MESSAGE_TYPE_MESSAGE
COMMAND_TYPE = data_model.MESSAGE_TYPE_COMMAND

MESSAGE_1 = {'message_id': MESSAGE_ID_1, 'type': MESSAGE_TYPE}
MESSAGE_2 = {'message_id': MESSAGE_ID_2, 'type': MESSAGE_TYPE}
COMMAND_1 = {'message_id': COMMAND_ID_1, 'type': COMMAND_TYPE}

AGENT_ID = 'AGENT_ID'

ACT_1 = {'text': 'THIS IS A MESSAGE', 'id': AGENT_ID}
ACT_2 = {'text': 'THIS IS A MESSAGE AGAIN', 'id': AGENT_ID}

active_statuses = [
    AssignState.STATUS_NONE,
    AssignState.STATUS_ONBOARDING,
    AssignState.STATUS_WAITING,
    AssignState.STATUS_IN_TASK,
]
complete_statuses = [
    AssignState.STATUS_DONE,
    AssignState.STATUS_DISCONNECT,
    AssignState.STATUS_PARTNER_DISCONNECT,
    AssignState.STATUS_PARTNER_DISCONNECT_EARLY,
    AssignState.STATUS_EXPIRED,
    AssignState.STATUS_RETURNED,
]
statuses = active_statuses + complete_statuses


class TestAssignState(unittest.TestCase):
    """
    Various unit tests for the AssignState class.
    """

    def setUp(self):
        self.agent_state1 = AssignState()
        self.agent_state2 = AssignState(status=AssignState.STATUS_IN_TASK)
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args([], print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        mturk_agent_ids = ['mturk_agent_1']
        self.mturk_manager = MTurkManager(opt=self.opt, mturk_agent_ids=mturk_agent_ids)
        self.worker_manager = self.mturk_manager.worker_manager

    def tearDown(self):
        self.mturk_manager.shutdown()

    def test_assign_state_init(self):
        """
        Test proper initialization of assignment states.
        """
        self.assertEqual(self.agent_state1.status, AssignState.STATUS_NONE)
        self.assertEqual(len(self.agent_state1.messages), 0)
        self.assertEqual(len(self.agent_state1.message_ids), 0)
        self.assertIsNone(self.agent_state1.last_command)
        self.assertEqual(self.agent_state2.status, AssignState.STATUS_IN_TASK)
        self.assertEqual(len(self.agent_state1.messages), 0)
        self.assertEqual(len(self.agent_state1.message_ids), 0)
        self.assertIsNone(self.agent_state1.last_command)

    def test_message_management(self):
        """
        Test message management in an AssignState.
        """
        # Ensure message appends succeed and are idempotent
        self.agent_state1.append_message(MESSAGE_1)
        self.assertEqual(len(self.agent_state1.get_messages()), 1)
        self.agent_state1.append_message(MESSAGE_2)
        self.assertEqual(len(self.agent_state1.get_messages()), 2)
        self.agent_state1.append_message(MESSAGE_1)
        self.assertEqual(len(self.agent_state1.get_messages()), 2)
        self.assertEqual(len(self.agent_state2.get_messages()), 0)
        self.assertIn(MESSAGE_1, self.agent_state1.get_messages())
        self.assertIn(MESSAGE_2, self.agent_state1.get_messages())
        self.assertEqual(len(self.agent_state1.message_ids), 2)
        self.agent_state2.append_message(MESSAGE_1)
        self.assertEqual(len(self.agent_state2.message_ids), 1)

        # Ensure command interactions work as expected
        self.agent_state1.set_last_command(COMMAND_1)
        self.assertEqual(self.agent_state1.get_last_command(), COMMAND_1)

        # Ensure clearing messages acts as intended and doesn't clear agent2
        self.agent_state1.clear_messages()
        self.assertEqual(len(self.agent_state1.messages), 0)
        self.assertEqual(len(self.agent_state1.message_ids), 0)
        self.assertIsNone(self.agent_state1.last_command)
        self.assertEqual(len(self.agent_state2.message_ids), 1)

    def test_state_handles_status(self):
        """
        Ensures status updates and is_final are valid.
        """

        for status in statuses:
            self.agent_state1.set_status(status)
            self.assertEqual(self.agent_state1.get_status(), status)

        for status in active_statuses:
            self.agent_state1.set_status(status)
            self.assertFalse(self.agent_state1.is_final())

        for status in complete_statuses:
            self.agent_state1.set_status(status)
            self.assertTrue(self.agent_state1.is_final())

        # TODO update the below once bonus is default
        for status in complete_statuses:
            self.agent_state1.set_status(status)
            text, command = self.agent_state1.get_inactive_command_text()
            self.assertIsNotNone(text)
            self.assertIsNotNone(command)


class TestMTurkAgent(unittest.TestCase):
    """
    Various unit tests for the MTurkAgent class.
    """

    def setUp(self):
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args([], print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        mturk_agent_ids = ['mturk_agent_1']
        self.mturk_manager = MTurkManager(
            opt=self.opt.copy(), mturk_agent_ids=mturk_agent_ids
        )
        self.worker_manager = self.mturk_manager.worker_manager

        self.turk_agent = MTurkAgent(
            self.opt.copy(),
            self.mturk_manager,
            TEST_HIT_ID_1,
            TEST_ASSIGNMENT_ID_1,
            TEST_WORKER_ID_1,
        )

    def tearDown(self):
        self.mturk_manager.shutdown()

        disconnect_path = os.path.join(parent_dir, 'disconnect-test.pickle')
        if os.path.exists(disconnect_path):
            os.remove(disconnect_path)

    def test_init(self):
        """
        Test initialization of an agent.
        """
        self.assertIsNotNone(self.turk_agent.creation_time)
        self.assertIsNone(self.turk_agent.id)
        self.assertIsNone(self.turk_agent.message_request_time)
        self.assertIsNone(self.turk_agent.conversation_id)
        self.assertFalse(self.turk_agent.some_agent_disconnected)
        self.assertFalse(self.turk_agent.hit_is_expired)
        self.assertFalse(self.turk_agent.hit_is_abandoned)
        self.assertFalse(self.turk_agent.hit_is_returned)
        self.assertFalse(self.turk_agent.hit_is_complete)
        self.assertFalse(self.turk_agent.disconnected)
        self.assertTrue(self.turk_agent.alived)

    def test_state_wrappers(self):
        """
        Test the mturk agent wrappers around its state.
        """
        for status in statuses:
            self.turk_agent.set_status(status)
            self.assertEqual(self.turk_agent.get_status(), status)
        for status in [AssignState.STATUS_DONE, AssignState.STATUS_PARTNER_DISCONNECT]:
            self.turk_agent.set_status(status)
            self.assertTrue(self.turk_agent.submitted_hit())

        for status in active_statuses:
            self.turk_agent.set_status(status)
            self.assertFalse(self.turk_agent.is_final())

        for status in complete_statuses:
            self.turk_agent.set_status(status)
            self.assertTrue(self.turk_agent.is_final())

        self.turk_agent.append_message(MESSAGE_1)
        self.assertEqual(len(self.turk_agent.get_messages()), 1)
        self.turk_agent.append_message(MESSAGE_2)
        self.assertEqual(len(self.turk_agent.get_messages()), 2)
        self.turk_agent.append_message(MESSAGE_1)
        self.assertEqual(len(self.turk_agent.get_messages()), 2)
        self.assertIn(MESSAGE_1, self.turk_agent.get_messages())
        self.assertIn(MESSAGE_2, self.turk_agent.get_messages())

        # Ensure command interactions work as expected
        self.turk_agent.set_last_command(COMMAND_1)
        self.assertEqual(self.turk_agent.get_last_command(), COMMAND_1)

        self.turk_agent.clear_messages()
        self.assertEqual(len(self.turk_agent.get_messages()), 0)

        # In task checks
        self.turk_agent.conversation_id = 't_12345'
        self.assertTrue(self.turk_agent.is_in_task())
        self.turk_agent.conversation_id = 'b_12345'
        self.assertFalse(self.turk_agent.is_in_task())

    def test_connection_id(self):
        """
        Ensure the connection_id hasn't changed.
        """
        connection_id = "{}_{}".format(
            self.turk_agent.worker_id, self.turk_agent.assignment_id
        )
        self.assertEqual(self.turk_agent.get_connection_id(), connection_id)

    def test_inactive_data(self):
        """
        Ensure data packet generated for inactive commands is valid.
        """
        for status in complete_statuses:
            self.turk_agent.set_status(status)
            data = self.turk_agent.get_inactive_command_data()
            self.assertIsNotNone(data['text'])
            self.assertIsNotNone(data['inactive_text'])
            self.assertEqual(data['conversation_id'], self.turk_agent.conversation_id)
            self.assertEqual(data['agent_id'], TEST_WORKER_ID_1)

    def test_status_change(self):
        has_changed = False
        self.turk_agent.set_status(AssignState.STATUS_ONBOARDING)

        def wait_for_status_wrap():
            nonlocal has_changed  # noqa 999 we don't use python2
            self.turk_agent.wait_for_status(AssignState.STATUS_WAITING)
            has_changed = True

        t = threading.Thread(target=wait_for_status_wrap, daemon=True)
        t.start()
        self.assertFalse(has_changed)
        time.sleep(0.07)
        self.assertFalse(has_changed)
        self.turk_agent.set_status(AssignState.STATUS_WAITING)
        time.sleep(0.07)
        self.assertTrue(has_changed)

    def test_message_queue(self):
        """
        Ensure observations and acts work as expected.
        """
        self.mturk_manager.send_message = mock.MagicMock()
        self.turk_agent.observe(ACT_1)
        self.mturk_manager.send_message.assert_called_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1, ACT_1
        )

        # First act comes through the queue and returns properly
        self.assertTrue(self.turk_agent.msg_queue.empty())
        self.turk_agent.id = AGENT_ID
        self.turk_agent.put_data(MESSAGE_ID_1, ACT_1)
        self.assertTrue(self.turk_agent.recieved_packets[MESSAGE_ID_1])
        self.assertFalse(self.turk_agent.msg_queue.empty())
        returned_act = self.turk_agent.get_new_act_message()
        self.assertEqual(returned_act, ACT_1)

        # Repeat act is ignored
        self.turk_agent.put_data(MESSAGE_ID_1, ACT_1)
        self.assertTrue(self.turk_agent.msg_queue.empty())

        for i in range(100):
            self.turk_agent.put_data(str(i), ACT_1)
        self.assertEqual(self.turk_agent.msg_queue.qsize(), 100)
        self.turk_agent.flush_msg_queue()
        self.assertTrue(self.turk_agent.msg_queue.empty())

        # Test non-act messages
        blank_message = self.turk_agent.get_new_act_message()
        self.assertIsNone(blank_message)

        self.turk_agent.disconnected = True
        disconnect_message = self.turk_agent.get_new_act_message()
        self.turk_agent.disconnected = False
        self.assertEqual(
            disconnect_message['text'], self.turk_agent.MTURK_DISCONNECT_MESSAGE
        )
        self.turk_agent.hit_is_returned = True
        return_message = self.turk_agent.get_new_act_message()
        self.assertEqual(return_message['text'], self.turk_agent.RETURN_MESSAGE)
        self.turk_agent.hit_is_returned = False

        # Reduce state
        self.turk_agent.reduce_state()
        self.assertIsNone(self.turk_agent.msg_queue)
        self.assertIsNone(self.turk_agent.recieved_packets)

    def test_message_acts(self):
        self.mturk_manager.send_command = mock.MagicMock()
        self.mturk_manager.handle_turker_timeout = mock.MagicMock()

        # non-Blocking check
        self.assertIsNone(self.turk_agent.message_request_time)
        returned_act = self.turk_agent.act(blocking=False)
        self.assertIsNotNone(self.turk_agent.message_request_time)
        self.assertIsNone(returned_act)
        self.turk_agent.id = AGENT_ID
        self.turk_agent.put_data(MESSAGE_ID_1, ACT_1)
        returned_act = self.turk_agent.act(blocking=False)
        self.assertIsNone(self.turk_agent.message_request_time)
        self.assertEqual(returned_act, ACT_1)
        self.mturk_manager.send_command.assert_called_once()

        # non-Blocking timeout check
        self.mturk_manager.send_command = mock.MagicMock()
        returned_act = self.turk_agent.act(timeout=0.07, blocking=False)
        self.assertIsNotNone(self.turk_agent.message_request_time)
        self.assertIsNone(returned_act)
        while returned_act is None:
            returned_act = self.turk_agent.act(timeout=0.07, blocking=False)
        self.mturk_manager.send_command.assert_called_once()
        self.mturk_manager.handle_turker_timeout.assert_called_once()
        self.assertEqual(returned_act['text'], self.turk_agent.TIMEOUT_MESSAGE)

        # Blocking timeout check
        self.mturk_manager.send_command = mock.MagicMock()
        self.mturk_manager.handle_turker_timeout = mock.MagicMock()
        returned_act = self.turk_agent.act(timeout=0.07)
        self.mturk_manager.send_command.assert_called_once()
        self.mturk_manager.handle_turker_timeout.assert_called_once()
        self.assertEqual(returned_act['text'], self.turk_agent.TIMEOUT_MESSAGE)


if __name__ == '__main__':
    unittest.main(buffer=True)
