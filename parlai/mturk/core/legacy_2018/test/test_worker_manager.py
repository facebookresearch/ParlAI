#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import os
from unittest import mock
from parlai.mturk.core.worker_manager import WorkerManager, WorkerState
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
TEST_WORKER_ID_2 = 'TEST_WORKER_ID_2'
TEST_WORKER_ID_3 = 'TEST_WORKER_ID_3'
TEST_ASSIGNMENT_ID_1 = 'TEST_ASSIGNMENT_ID_1'
TEST_ASSIGNMENT_ID_2 = 'TEST_ASSIGNMENT_ID_2'
TEST_ASSIGNMENT_ID_3 = 'TEST_ASSIGNMENT_ID_3'
TEST_HIT_ID_1 = 'TEST_HIT_ID_1'
TEST_HIT_ID_2 = 'TEST_HIT_ID_2'
TEST_HIT_ID_3 = 'TEST_HIT_ID_3'
FAKE_ID = 'BOGUS'


class TestWorkerState(unittest.TestCase):
    """
    Various unit tests for the WorkerState class.
    """

    def setUp(self):
        self.work_state_1 = WorkerState(TEST_WORKER_ID_1, 10)
        self.work_state_2 = WorkerState(TEST_WORKER_ID_2)
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args(print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        mturk_agent_ids = ['mturk_agent_1']
        self.mturk_manager = MTurkManager(opt=self.opt, mturk_agent_ids=mturk_agent_ids)
        self.worker_manager = WorkerManager(self.mturk_manager, self.opt)

    def tearDown(self):
        self.mturk_manager.shutdown()

    def test_worker_state_init(self):
        """
        Test proper initialization of worker states.
        """
        self.assertEqual(self.work_state_1.worker_id, TEST_WORKER_ID_1)
        self.assertEqual(self.work_state_2.worker_id, TEST_WORKER_ID_2)
        self.assertEqual(self.work_state_1.disconnects, 10)
        self.assertEqual(self.work_state_2.disconnects, 0)

    def test_worker_state_agent_management(self):
        """
        Test public state management methods of worker_state.
        """
        agent_1 = MTurkAgent(
            self.opt,
            self.mturk_manager,
            TEST_HIT_ID_1,
            TEST_ASSIGNMENT_ID_1,
            TEST_WORKER_ID_1,
        )
        agent_2 = MTurkAgent(
            self.opt,
            self.mturk_manager,
            TEST_HIT_ID_2,
            TEST_ASSIGNMENT_ID_2,
            TEST_WORKER_ID_1,
        )
        agent_3 = MTurkAgent(
            self.opt,
            self.mturk_manager,
            TEST_HIT_ID_3,
            TEST_ASSIGNMENT_ID_3,
            TEST_WORKER_ID_3,
        )

        self.assertEqual(self.work_state_1.active_conversation_count(), 0)
        self.work_state_1.add_agent(agent_1)
        self.assertEqual(self.work_state_1.active_conversation_count(), 1)
        self.work_state_1.add_agent(agent_2)
        self.assertEqual(self.work_state_1.active_conversation_count(), 2)

        with self.assertRaises(AssertionError):
            self.work_state_1.add_agent(agent_3)

        self.assertEqual(self.work_state_1.active_conversation_count(), 2)
        self.assertEqual(self.work_state_1.completed_assignments(), 0)

        self.assertTrue(self.work_state_1.has_assignment(agent_1.assignment_id))
        self.assertTrue(self.work_state_1.has_assignment(agent_2.assignment_id))
        self.assertFalse(self.work_state_1.has_assignment(agent_3.assignment_id))
        self.assertEqual(
            agent_1, self.work_state_1.get_agent_for_assignment(agent_1.assignment_id)
        )
        self.assertEqual(
            agent_2, self.work_state_1.get_agent_for_assignment(agent_2.assignment_id)
        )
        self.assertIsNone(
            self.work_state_1.get_agent_for_assignment(agent_3.assignment_id)
        )

        agent_1.set_status(AssignState.STATUS_DONE)
        self.assertEqual(self.work_state_1.active_conversation_count(), 1)
        self.assertEqual(self.work_state_1.completed_assignments(), 1)
        agent_2.set_status(AssignState.STATUS_DISCONNECT)
        self.assertEqual(self.work_state_1.active_conversation_count(), 0)
        self.assertEqual(self.work_state_1.completed_assignments(), 1)

    def test_manager_alive_makes_state(self):
        test_worker = self.worker_manager.worker_alive(TEST_WORKER_ID_1)
        self.assertIsInstance(test_worker, WorkerState)
        self.assertEqual(test_worker.worker_id, TEST_WORKER_ID_1)
        self.assertNotEqual(test_worker, self.work_state_1)


class TestWorkerManager(unittest.TestCase):
    """
    Various unit tests for the WorkerManager class.
    """

    def setUp(self):
        disconnect_path = os.path.join(parent_dir, 'disconnect-test.pickle')
        if os.path.exists(disconnect_path):
            os.remove(disconnect_path)

        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args(print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        mturk_agent_ids = ['mturk_agent_1']
        self.mturk_manager = MTurkManager(
            opt=self.opt.copy(), mturk_agent_ids=mturk_agent_ids
        )
        self.worker_manager = self.mturk_manager.worker_manager

        self.worker_state_1 = self.worker_manager.worker_alive(TEST_WORKER_ID_1)
        self.worker_state_2 = self.worker_manager.worker_alive(TEST_WORKER_ID_2)
        self.worker_state_3 = self.worker_manager.worker_alive(TEST_WORKER_ID_3)

    def tearDown(self):
        self.mturk_manager.shutdown()

        disconnect_path = os.path.join(parent_dir, 'disconnect-test.pickle')
        if os.path.exists(disconnect_path):
            os.remove(disconnect_path)

    def test_private_create_agent(self):
        """
        Check create agent method used internally in worker_manager.
        """
        test_agent = self.worker_manager._create_agent(
            TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1
        )
        self.assertIsInstance(test_agent, MTurkAgent)
        self.assertEqual(test_agent.worker_id, TEST_WORKER_ID_1)
        self.assertEqual(test_agent.hit_id, TEST_HIT_ID_1)
        self.assertEqual(test_agent.assignment_id, TEST_ASSIGNMENT_ID_1)

    def test_agent_task_management(self):
        """
        Ensure agents and tasks have proper bookkeeping.
        """
        self.worker_manager.assign_task_to_worker(
            TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1
        )
        self.worker_manager.assign_task_to_worker(
            TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2
        )
        self.worker_manager.assign_task_to_worker(
            TEST_HIT_ID_3, TEST_ASSIGNMENT_ID_3, TEST_WORKER_ID_1
        )
        self.assertTrue(self.worker_state_1.has_assignment(TEST_ASSIGNMENT_ID_1))
        self.assertTrue(self.worker_state_1.has_assignment(TEST_ASSIGNMENT_ID_3))
        self.assertTrue(self.worker_state_2.has_assignment(TEST_ASSIGNMENT_ID_2))

        assign_agent = self.worker_manager.get_agent_for_assignment(
            TEST_ASSIGNMENT_ID_1
        )
        self.assertEqual(assign_agent.worker_id, TEST_WORKER_ID_1)
        self.assertEqual(assign_agent.hit_id, TEST_HIT_ID_1)
        self.assertEqual(assign_agent.assignment_id, TEST_ASSIGNMENT_ID_1)

        no_such_agent = self.worker_manager.get_agent_for_assignment(FAKE_ID)
        self.assertIsNone(no_such_agent)

        # Ensure all agents are being maintained
        checked_count = 0
        filtered_count = 0

        def check_is_worker_1(agent):
            nonlocal checked_count  # noqa E999 python 3 only
            checked_count += 1
            self.assertEqual(agent.worker_id, TEST_WORKER_ID_1)

        def is_worker_1(agent):
            nonlocal filtered_count
            filtered_count += 1
            return agent.worker_id == TEST_WORKER_ID_1

        self.worker_manager.map_over_agents(check_is_worker_1, is_worker_1)
        self.assertEqual(checked_count, 2)
        self.assertEqual(filtered_count, 3)

        # Ensuring _get_worker is accurate
        self.assertEqual(
            self.worker_manager._get_worker(TEST_WORKER_ID_1), self.worker_state_1
        )
        self.assertEqual(
            self.worker_manager._get_worker(TEST_WORKER_ID_2), self.worker_state_2
        )
        self.assertEqual(
            self.worker_manager._get_worker(TEST_WORKER_ID_3), self.worker_state_3
        )
        self.assertIsNone(self.worker_manager._get_worker(FAKE_ID))

        # Ensuring _get_agent is accurate
        self.assertEqual(
            self.worker_manager._get_agent(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1),
            self.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_1),
        )
        self.assertNotEqual(
            self.worker_manager._get_agent(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_2),
            self.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_2),
        )

    def test_shutdown(self):
        """
        Ensure shutdown clears required resources.
        """
        self.worker_manager.save_disconnects = mock.MagicMock()
        self.worker_manager.un_time_block_workers = mock.MagicMock()
        self.worker_manager.shutdown()
        self.assertEqual(
            len(self.worker_manager.save_disconnects.mock_calls),
            1,
            'save_disconnects must be called in worker manager shutdown',
        )
        self.assertEqual(
            len(self.worker_manager.un_time_block_workers.mock_calls),
            1,
            'un_time_block_workers must be called in worker manager shutdown',
        )

    def test_time_blocks(self):
        """
        Check to see if time blocking and clearing works.
        """
        self.mturk_manager.soft_block_worker = mock.MagicMock()
        self.mturk_manager.un_soft_block_worker = mock.MagicMock()

        # No workers blocked, none should be unblocked
        self.worker_manager.un_time_block_workers()
        self.assertEqual(len(self.mturk_manager.un_soft_block_worker.mock_calls), 0)

        # Block some workers, ensure state change and correct calls
        self.assertEqual(len(self.worker_manager.time_blocked_workers), 0)
        self.worker_manager.time_block_worker(TEST_WORKER_ID_1)
        self.mturk_manager.soft_block_worker.assert_called_with(
            TEST_WORKER_ID_1, 'max_time_qual'
        )
        self.assertEqual(len(self.worker_manager.time_blocked_workers), 1)
        self.worker_manager.time_block_worker(TEST_WORKER_ID_2)
        self.mturk_manager.soft_block_worker.assert_called_with(
            TEST_WORKER_ID_2, 'max_time_qual'
        )
        self.assertEqual(len(self.worker_manager.time_blocked_workers), 2)
        self.assertEqual(len(self.mturk_manager.soft_block_worker.mock_calls), 2)

        # Unblock a worker passed in as a keyword arg, ensure state remains
        self.worker_manager.un_time_block_workers([TEST_WORKER_ID_3])
        self.mturk_manager.un_soft_block_worker.assert_called_with(
            TEST_WORKER_ID_3, 'max_time_qual'
        )
        self.assertEqual(len(self.worker_manager.time_blocked_workers), 2)

        # Unblock blocked workers, ensure proper calls and state change
        self.worker_manager.un_time_block_workers()
        self.assertEqual(len(self.worker_manager.time_blocked_workers), 0)
        self.mturk_manager.un_soft_block_worker.assert_any_call(
            TEST_WORKER_ID_1, 'max_time_qual'
        )
        self.mturk_manager.un_soft_block_worker.assert_any_call(
            TEST_WORKER_ID_2, 'max_time_qual'
        )

    def test_disconnect_management(self):
        self.worker_manager.load_disconnects()
        self.worker_manager.is_sandbox = False
        self.mturk_manager.block_worker = mock.MagicMock()
        self.mturk_manager.soft_block_worker = mock.MagicMock()

        self.assertEqual(len(self.worker_manager.disconnects), 0)
        # Make one worker disconnect twice
        self.worker_manager.handle_bad_disconnect(TEST_WORKER_ID_1)
        self.assertEqual(len(self.worker_manager.disconnects), 1)
        self.mturk_manager.block_worker.assert_not_called()
        self.mturk_manager.soft_block_worker.assert_not_called()
        self.worker_manager.handle_bad_disconnect(TEST_WORKER_ID_1)
        self.assertEqual(len(self.worker_manager.disconnects), 2)
        self.mturk_manager.block_worker.assert_not_called()
        self.mturk_manager.soft_block_worker.assert_not_called()

        # Ensure both disconnects recorded
        self.assertEqual(
            self.worker_manager.mturk_workers[TEST_WORKER_ID_1].disconnects, 2
        )

        # Make second worker disconnect
        self.worker_manager.handle_bad_disconnect(TEST_WORKER_ID_2)
        self.assertEqual(len(self.worker_manager.disconnects), 3)
        self.mturk_manager.block_worker.assert_not_called()
        self.mturk_manager.soft_block_worker.assert_not_called()

        # Make us soft block workers on disconnect
        self.worker_manager.opt['disconnect_qualification'] = 'test'
        self.worker_manager.handle_bad_disconnect(TEST_WORKER_ID_1)
        self.mturk_manager.block_worker.assert_not_called()
        self.mturk_manager.soft_block_worker.assert_called_with(
            TEST_WORKER_ID_1, 'disconnect_qualification'
        )
        self.mturk_manager.soft_block_worker.reset_mock()

        # Make us now block workers on disconnect
        self.worker_manager.opt['hard_block'] = True
        self.worker_manager.handle_bad_disconnect(TEST_WORKER_ID_2)
        self.mturk_manager.block_worker.assert_called_once()
        self.mturk_manager.soft_block_worker.assert_not_called()

        # Ensure we can save and reload disconnects
        self.worker_manager.save_disconnects()

        # Make a new worker manager
        worker_manager2 = WorkerManager(self.mturk_manager, self.opt)
        self.assertEqual(len(worker_manager2.disconnects), 5)
        self.assertEqual(worker_manager2.mturk_workers[TEST_WORKER_ID_1].disconnects, 3)
        self.assertEqual(worker_manager2.mturk_workers[TEST_WORKER_ID_2].disconnects, 2)
        worker_manager2.shutdown()

    def test_conversation_management(self):
        """
        Tests handling conversation state, moving agents to the correct conversations,
        and disconnecting one worker in an active convo.
        """
        self.worker_manager.assign_task_to_worker(
            TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1
        )
        self.worker_manager.assign_task_to_worker(
            TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2
        )

        good_agent = self.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_1)
        bad_agent = self.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_2)

        def fake_command_send(worker_id, assignment_id, data, ack_func):
            pkt = mock.MagicMock()
            pkt.sender_id = worker_id
            pkt.assignment_id = assignment_id
            self.assertEqual(data['text'], data_model.COMMAND_CHANGE_CONVERSATION)
            ack_func(pkt)

        self.mturk_manager.send_command = fake_command_send
        self.worker_manager.change_agent_conversation(good_agent, 't1', 'good')
        self.worker_manager.change_agent_conversation(bad_agent, 't1', 'bad')
        self.assertEqual(good_agent.id, 'good')
        self.assertEqual(bad_agent.id, 'bad')
        self.assertEqual(good_agent.conversation_id, 't1')
        self.assertEqual(bad_agent.conversation_id, 't1')
        self.assertIn('t1', self.worker_manager.conv_to_agent)
        self.assertEqual(len(self.worker_manager.conv_to_agent['t1']), 2)
        self.worker_manager.handle_bad_disconnect = mock.MagicMock()

        checked_worker = False

        def partner_callback(agent):
            nonlocal checked_worker
            checked_worker = True
            self.assertEqual(agent.worker_id, good_agent.worker_id)

        self.worker_manager.handle_agent_disconnect(
            bad_agent.worker_id, bad_agent.assignment_id, partner_callback
        )
        self.assertTrue(checked_worker)
        self.worker_manager.handle_bad_disconnect.assert_called_once_with(
            bad_agent.worker_id
        )
        self.assertEqual(bad_agent.get_status(), AssignState.STATUS_DISCONNECT)


if __name__ == '__main__':
    unittest.main(buffer=True)
