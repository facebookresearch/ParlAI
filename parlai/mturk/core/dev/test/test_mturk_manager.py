#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import os
import time
import json
import threading
import pickle
from unittest import mock
from parlai.mturk.core.worker_manager import WorkerManager
from parlai.mturk.core.agents import MTurkAgent, AssignState
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.core.socket_manager import SocketManager, Packet
from parlai.core.params import ParlaiParser
from websocket_server import WebsocketServer

import parlai.mturk.core.mturk_manager as MTurkManagerFile
import parlai.mturk.core.data_model as data_model

parent_dir = os.path.dirname(os.path.abspath(__file__))
MTurkManagerFile.parent_dir = os.path.dirname(os.path.abspath(__file__))

MTurkManagerFile.mturk_utils = mock.MagicMock()

# Lets ignore the logging part
MTurkManagerFile.shared_utils.print_and_log = mock.MagicMock()

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


def assert_equal_by(val_func, val, max_time):
    start_time = time.time()
    while val_func() != val:
        assert time.time() - start_time < max_time, \
            "Value was not attained in specified time"
        time.sleep(0.1)


class MockSocket():
    def __init__(self):
        self.last_messages = {}
        self.connected = False
        self.disconnected = False
        self.closed = False
        self.ws = None
        self.should_heartbeat = True
        self.fake_workers = []
        self.port = None
        self.launch_socket()
        self.handlers = {}
        while self.ws is None:
            time.sleep(0.05)
        time.sleep(1)

    def send(self, packet):
        self.ws.send_message_to_all(packet)

    def close(self):
        if not self.closed:
            self.ws.server_close()
            self.ws.shutdown()
            self.closed = True

    def do_nothing(self, *args):
        pass

    def launch_socket(self):
        def on_message(client, server, message):
            if self.closed:
                raise Exception('Socket is already closed...')
            if message == '':
                return
            packet_dict = json.loads(message)
            if packet_dict['content']['id'] == 'WORLD_ALIVE':
                self.ws.send_message(
                    client, json.dumps({'type': 'conn_success'}))
                self.connected = True
            elif packet_dict['content']['type'] == 'heartbeat':
                pong = packet_dict['content'].copy()
                pong['type'] = 'pong'
                self.ws.send_message(client, json.dumps({
                    'type': data_model.SOCKET_ROUTE_PACKET_STRING,
                    'content': pong,
                }))
            if 'receiver_id' in packet_dict['content']:
                receiver_id = packet_dict['content']['receiver_id']
                use_func = self.handlers.get(receiver_id, self.do_nothing)
                use_func(packet_dict['content'])

        def on_connect(client, server):
            pass

        def on_disconnect(client, server):
            self.disconnected = True

        def run_socket(*args):
            port = 3030
            while self.port is None:
                try:
                    self.ws = WebsocketServer(port, host='127.0.0.1')
                    self.port = port
                except OSError:
                    port += 1
            self.ws.set_fn_client_left(on_disconnect)
            self.ws.set_fn_new_client(on_connect)
            self.ws.set_fn_message_received(on_message)
            self.ws.run_forever()

        self.listen_thread = threading.Thread(
            target=run_socket,
            name='Fake-Socket-Thread'
        )
        self.listen_thread.daemon = True
        self.listen_thread.start()


class InitTestMTurkManager(unittest.TestCase):
    '''Unit tests for MTurkManager setup'''
    def setUp(self):
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args(print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(
            opt=self.opt,
            mturk_agent_ids=self.mturk_agent_ids,
            is_test=True,
        )

    def tearDown(self):
        self.mturk_manager.shutdown()

    def test_init(self):
        manager = self.mturk_manager
        opt = self.opt
        self.assertIsNone(manager.server_url)
        self.assertIsNone(manager.topic_arn)
        self.assertIsNone(manager.server_task_name)
        self.assertIsNone(manager.task_group_id)
        self.assertIsNone(manager.run_id)
        self.assertIsNone(manager.task_files_to_copy)
        self.assertIsNone(manager.onboard_function)
        self.assertIsNone(manager.socket_manager)
        self.assertFalse(manager.is_shutdown)
        self.assertFalse(manager.is_unique)
        self.assertEqual(manager.opt, opt)
        self.assertEqual(manager.mturk_agent_ids,
                         self.mturk_agent_ids)
        self.assertEqual(manager.is_sandbox, opt['is_sandbox'])
        self.assertEqual(manager.num_conversations, opt['num_conversations'])
        self.assertEqual(manager.is_sandbox, opt['is_sandbox'])

        self.assertGreaterEqual(
            manager.required_hits,
            manager.num_conversations * len(self.mturk_agent_ids))

        self.assertIsNotNone(manager.agent_pool_change_condition)

        self.assertEqual(manager.minimum_messages, opt.get('min_messages', 0))
        self.assertEqual(manager.auto_approve_delay,
                         opt.get('auto_approve_delay', 4 * 7 * 24 * 3600))
        self.assertEqual(manager.has_time_limit,
                         opt.get('max_time', 0) > 0)
        self.assertIsInstance(manager.worker_manager, WorkerManager)
        self.assertEqual(manager.task_state, manager.STATE_CREATED)

    def test_init_state(self):
        manager = self.mturk_manager
        manager._init_state()
        self.assertEqual(manager.agent_pool, [])
        self.assertEqual(manager.hit_id_list, [])
        self.assertEqual(manager.conversation_index, 0)
        self.assertEqual(manager.started_conversations, 0)
        self.assertEqual(manager.completed_conversations, 0)
        self.assertEqual(manager.task_threads, [])
        self.assertTrue(manager.accepting_workers, True)
        self.assertIsNone(manager.qualifications)
        self.assertGreater(manager.time_limit_checked, time.time() - 1)
        self.assertEqual(manager.task_state, manager.STATE_INIT_RUN)


class TestMTurkManagerUnitFunctions(unittest.TestCase):
    '''Tests some of the simpler MTurkManager functions that don't require
    much additional state to run'''
    def setUp(self):
        self.fake_socket = MockSocket()
        time.sleep(0.1)
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args(print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(
            opt=self.opt,
            mturk_agent_ids=self.mturk_agent_ids,
            is_test=True,
        )
        self.mturk_manager._init_state()
        self.mturk_manager.port = self.fake_socket.port
        self.agent_1 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_1,
                                  TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1)
        self.agent_2 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_2,
                                  TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2)
        self.agent_3 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_3,
                                  TEST_ASSIGNMENT_ID_3, TEST_WORKER_ID_3)

    def tearDown(self):
        self.mturk_manager.shutdown()
        self.fake_socket.close()

    def test_move_to_waiting(self):
        manager = self.mturk_manager
        manager.worker_manager.change_agent_conversation = mock.MagicMock()
        manager.socket_manager = mock.MagicMock()
        manager.socket_manager.close_channel = mock.MagicMock()
        manager.force_expire_hit = mock.MagicMock()
        self.agent_1.set_status(AssignState.STATUS_DISCONNECT)
        self.agent_1.reduce_state = mock.MagicMock()
        self.agent_2.reduce_state = mock.MagicMock()
        self.agent_3.reduce_state = mock.MagicMock()

        # Test with a disconnected agent, assert the channel is closed
        manager._move_agents_to_waiting([self.agent_1])
        self.agent_1.reduce_state.assert_called_once()
        manager.socket_manager.close_channel.assert_called_once_with(
            self.agent_1.get_connection_id())
        manager.worker_manager.change_agent_conversation.assert_not_called()
        manager.force_expire_hit.assert_not_called()
        manager.socket_manager.close_channel.reset_mock()

        # Test with a connected agent, should be moved to waiting
        manager._move_agents_to_waiting([self.agent_2])
        self.agent_2.reduce_state.assert_not_called()
        manager.socket_manager.close_channel.assert_not_called()
        manager.worker_manager.change_agent_conversation.assert_called_once()
        args = manager.worker_manager.change_agent_conversation.call_args[1]
        self.assertEqual(args['agent'], self.agent_2)
        self.assertTrue(manager.is_waiting_world(args['conversation_id']))
        self.assertEqual(args['new_agent_id'], 'waiting')
        manager.force_expire_hit.assert_not_called()
        manager.worker_manager.change_agent_conversation.reset_mock()

        # Test when no longer accepting agents
        manager.accepting_workers = False
        manager._move_agents_to_waiting([self.agent_3])
        self.agent_3.reduce_state.assert_not_called()
        manager.socket_manager.close_channel.assert_not_called()
        manager.worker_manager.change_agent_conversation.assert_not_called()
        manager.force_expire_hit.assert_called_once_with(
            self.agent_3.worker_id, self.agent_3.assignment_id)

    def test_socket_setup(self):
        '''Basic socket setup should fail when not in correct state,
        but succeed otherwise
        '''
        self.mturk_manager.task_state = self.mturk_manager.STATE_CREATED
        with self.assertRaises(AssertionError):
            self.mturk_manager._setup_socket()
        self.mturk_manager.task_group_id = 'TEST_GROUP_ID'
        self.mturk_manager.server_url = 'https://127.0.0.1'
        self.mturk_manager.task_state = self.mturk_manager.STATE_INIT_RUN
        self.mturk_manager._setup_socket()
        self.assertIsInstance(self.mturk_manager.socket_manager, SocketManager)

    def test_worker_alive(self):
        # Setup for test
        manager = self.mturk_manager
        manager.task_group_id = 'TEST_GROUP_ID'
        manager.server_url = 'https://127.0.0.1'
        manager.task_state = manager.STATE_ACCEPTING_WORKERS
        manager._setup_socket()
        manager.force_expire_hit = mock.MagicMock()
        manager._onboard_new_agent = mock.MagicMock()
        manager.socket_manager.open_channel = \
            mock.MagicMock(wraps=manager.socket_manager.open_channel)
        manager.worker_manager.worker_alive = \
            mock.MagicMock(wraps=manager.worker_manager.worker_alive)
        open_channel = manager.socket_manager.open_channel
        worker_alive = manager.worker_manager.worker_alive

        # Test no assignment
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': None,
            'conversation_id': None,
        }, '')
        manager._on_alive(alive_packet)
        open_channel.assert_not_called()
        worker_alive.assert_not_called()
        manager._onboard_new_agent.assert_not_called()

        # Test not accepting workers
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_1,
            'conversation_id': None,
        }, '')
        manager.accepting_workers = False
        manager._on_alive(alive_packet)
        open_channel.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        worker_alive.assert_called_once_with(TEST_WORKER_ID_1)
        worker_state = manager.worker_manager._get_worker(TEST_WORKER_ID_1)
        self.assertIsNotNone(worker_state)
        open_channel.reset_mock()
        worker_alive.reset_mock()
        manager.force_expire_hit.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        manager._onboard_new_agent.assert_not_called()
        manager.force_expire_hit.reset_mock()

        # Test successful creation
        manager.accepting_workers = True
        manager._on_alive(alive_packet)
        open_channel.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        worker_alive.assert_called_once_with(TEST_WORKER_ID_1)
        manager._onboard_new_agent.assert_called_once()
        manager._onboard_new_agent.reset_mock()
        manager.force_expire_hit.assert_not_called()

        agent = manager.worker_manager.get_agent_for_assignment(
            TEST_ASSIGNMENT_ID_1)
        self.assertIsInstance(agent, MTurkAgent)
        self.assertEqual(agent.get_status(), AssignState.STATUS_NONE)

        # Reconnect in various conditions
        agent.set_status = mock.MagicMock(wraps=agent.set_status)
        manager._add_agent_to_pool = mock.MagicMock()

        # Reconnect when none state no connection_id
        agent.log_reconnect = mock.MagicMock(wraps=agent.log_reconnect)
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        manager.force_expire_hit.reset_mock()
        agent.set_status.assert_not_called()
        manager._add_agent_to_pool.assert_not_called()
        agent.log_reconnect.assert_called_once()
        agent.log_reconnect.reset_mock()

        # Reconnect in None state onboarding conversation_id
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_1,
            'conversation_id': 'o_1234',
        }, '')
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_not_called()
        agent.set_status.assert_called_once_with(AssignState.STATUS_ONBOARDING)
        manager._add_agent_to_pool.assert_not_called()
        agent.log_reconnect.assert_called_once()
        agent.log_reconnect.reset_mock()

        # Reconnect in None state waiting conversation_id
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_1,
            'conversation_id': 'w_1234',
        }, '')
        agent.set_status(AssignState.STATUS_NONE)
        agent.set_status.reset_mock()
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_not_called()
        agent.set_status.assert_called_once_with(AssignState.STATUS_WAITING)
        manager._add_agent_to_pool.assert_called_once_with(agent)
        manager._add_agent_to_pool.reset_mock()
        agent.log_reconnect.assert_called_once()
        agent.log_reconnect.reset_mock()

        # Reconnect in onboarding with waiting conversation id
        agent.set_status(AssignState.STATUS_ONBOARDING)
        agent.set_status.reset_mock()
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_not_called()
        agent.set_status.assert_called_once_with(AssignState.STATUS_WAITING)
        manager._add_agent_to_pool.assert_called_once_with(agent)
        manager._add_agent_to_pool.reset_mock()
        agent.log_reconnect.assert_called_once()
        agent.log_reconnect.reset_mock()

        # Reconnect in onboarding with no conversation id
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_1,
            'conversation_id': None,
        }, '')
        agent.set_status(AssignState.STATUS_ONBOARDING)
        agent.set_status.reset_mock()
        manager._restore_agent_state = mock.MagicMock()
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_not_called()
        manager._restore_agent_state.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        agent.set_status.assert_not_called()
        manager._add_agent_to_pool.assert_not_called()
        agent.log_reconnect.assert_called_once()
        agent.log_reconnect.reset_mock()

        # Reconnect in onboarding but not accepting new workers
        manager.accepting_workers = False
        agent.set_status(AssignState.STATUS_ONBOARDING)
        agent.set_status.reset_mock()
        manager._restore_agent_state = mock.MagicMock()
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        manager.force_expire_hit.reset_mock()
        manager._restore_agent_state.assert_not_called()
        agent.set_status.assert_not_called()
        manager._add_agent_to_pool.assert_not_called()
        agent.log_reconnect.assert_called_once()
        agent.log_reconnect.reset_mock()

        # Reconnect in waiting no conv id
        manager.accepting_workers = True
        agent.set_status(AssignState.STATUS_WAITING)
        agent.set_status.reset_mock()
        manager._restore_agent_state = mock.MagicMock()
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_not_called()
        manager._restore_agent_state.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        agent.set_status.assert_not_called()
        manager._add_agent_to_pool.assert_called_once()
        manager._add_agent_to_pool.reset_mock()
        agent.log_reconnect.assert_called_once()
        agent.log_reconnect.reset_mock()

        # Reconnect in waiting with conv id
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_1,
            'conversation_id': 'w_1234',
        }, '')
        agent.set_status(AssignState.STATUS_WAITING)
        agent.set_status.reset_mock()
        manager._restore_agent_state = mock.MagicMock()
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_not_called()
        manager._restore_agent_state.assert_not_called()
        agent.set_status.assert_not_called()
        manager._add_agent_to_pool.assert_called_once()
        manager._add_agent_to_pool.reset_mock()
        agent.log_reconnect.assert_called_once()
        agent.log_reconnect.reset_mock()

        # Reconnect in waiting with task id
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_1,
            'conversation_id': 't_1234',
        }, '')
        agent.set_status(AssignState.STATUS_WAITING)
        agent.set_status.reset_mock()
        manager._restore_agent_state = mock.MagicMock()
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_not_called()
        manager._restore_agent_state.assert_not_called()
        agent.set_status.assert_called_with(AssignState.STATUS_IN_TASK)
        manager._add_agent_to_pool.assert_not_called()
        agent.log_reconnect.assert_called_once()
        agent.log_reconnect.reset_mock()

        # Test active convos failure
        agent.set_status(AssignState.STATUS_IN_TASK)
        agent.set_status.reset_mock()
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_2,
            'conversation_id': None,
        }, '')
        manager.opt['allowed_conversations'] = 1
        manager._on_alive(alive_packet)
        agent.set_status.assert_not_called()
        manager.force_expire_hit.assert_called_once()
        manager._onboard_new_agent.assert_not_called()
        manager.force_expire_hit.reset_mock()

        # Test uniqueness failed
        agent.set_status(AssignState.STATUS_DONE)
        agent.set_status.reset_mock()
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_2,
            'conversation_id': None,
        }, '')
        manager.is_unique = True
        manager._on_alive(alive_packet)
        agent.set_status.assert_not_called()
        manager.force_expire_hit.assert_called_once()
        manager._onboard_new_agent.assert_not_called()
        manager.force_expire_hit.reset_mock()

        # Test in task reconnects
        agent.set_status(AssignState.STATUS_IN_TASK)
        agent.set_status.reset_mock()
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_1,
            'conversation_id': None,
        }, '')
        manager._on_alive(alive_packet)
        manager._restore_agent_state.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        agent.set_status.assert_not_called()
        manager._add_agent_to_pool.assert_not_called()
        agent.log_reconnect.assert_called_once()
        agent.log_reconnect.reset_mock()

        # Test all final states
        for use_state in [
            AssignState.STATUS_DISCONNECT, AssignState.STATUS_DONE,
            AssignState.STATUS_EXPIRED, AssignState.STATUS_RETURNED,
            AssignState.STATUS_PARTNER_DISCONNECT
        ]:
            manager.send_command = mock.MagicMock()
            agent.set_status(use_state)
            agent.set_status.reset_mock()
            manager._on_alive(alive_packet)
            agent.set_status.assert_not_called()
            manager._add_agent_to_pool.assert_not_called()
            manager.force_expire_hit.assert_not_called()
            manager.send_command.assert_called_once()

    def test_mturk_messages(self):
        '''Ensure incoming messages work as expected'''
        # Setup for test
        manager = self.mturk_manager
        manager.task_group_id = 'TEST_GROUP_ID'
        manager.server_url = 'https://127.0.0.1'
        manager.task_state = manager.STATE_ACCEPTING_WORKERS
        manager._setup_socket()
        manager.force_expire_hit = mock.MagicMock()
        manager._on_socket_dead = mock.MagicMock()

        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_1,
            'conversation_id': None,
        }, '')
        manager._on_alive(alive_packet)
        agent = manager.worker_manager.get_agent_for_assignment(
            TEST_ASSIGNMENT_ID_1)
        self.assertIsInstance(agent, MTurkAgent)
        self.assertEqual(agent.get_status(), AssignState.STATUS_NONE)
        agent.set_hit_is_abandoned = mock.MagicMock()

        # Test SNS_ASSIGN_ABANDONDED
        message_packet = Packet('', '', '', '', TEST_ASSIGNMENT_ID_1, {
            'text': MTurkManagerFile.SNS_ASSIGN_ABANDONDED,
        }, '')
        manager._handle_mturk_message(message_packet)
        agent.set_hit_is_abandoned.assert_called_once()
        agent.set_hit_is_abandoned.reset_mock()
        manager._on_socket_dead.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1
        )
        manager._on_socket_dead.reset_mock()

        # Test SNS_ASSIGN_RETURNED
        message_packet = Packet('', '', '', '', TEST_ASSIGNMENT_ID_1, {
            'text': MTurkManagerFile.SNS_ASSIGN_RETURNED,
        }, '')
        agent.hit_is_returned = False
        manager._handle_mturk_message(message_packet)
        manager._on_socket_dead.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1
        )
        manager._on_socket_dead.reset_mock()
        self.assertTrue(agent.hit_is_returned)

        # Test SNS_ASSIGN_SUBMITTED
        message_packet = Packet('', '', '', '', TEST_ASSIGNMENT_ID_1, {
            'text': MTurkManagerFile.SNS_ASSIGN_SUBMITTED,
        }, '')
        agent.hit_is_complete = False
        manager._handle_mturk_message(message_packet)
        manager._on_socket_dead.assert_not_called()
        self.assertTrue(agent.hit_is_complete)

    def test_new_message(self):
        '''test on_new_message'''
        alive_packet = Packet('', TEST_WORKER_ID_1, '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_1,
            'conversation_id': None,
        }, '')

        message_packet = Packet(
            '', '', MTurkManagerFile.AMAZON_SNS_NAME, '',
            TEST_ASSIGNMENT_ID_1, {
                'text': MTurkManagerFile.SNS_ASSIGN_SUBMITTED,
            }, '')

        manager = self.mturk_manager
        manager._handle_mturk_message = mock.MagicMock()
        manager.worker_manager.route_packet = mock.MagicMock()

        # test mturk message
        manager._on_new_message(alive_packet)
        manager._handle_mturk_message.assert_not_called()
        manager.worker_manager.route_packet.assert_called_once_with(
            alive_packet)
        manager.worker_manager.route_packet.reset_mock()

        # test non-mturk message
        manager._on_new_message(message_packet)
        manager._handle_mturk_message.assert_called_once_with(message_packet)
        manager.worker_manager.route_packet.assert_not_called()

    def test_onboarding_function(self):
        manager = self.mturk_manager
        manager.onboard_function = mock.MagicMock()
        manager.worker_manager.change_agent_conversation = mock.MagicMock()
        manager._move_agents_to_waiting = mock.MagicMock()
        manager.worker_manager.get_agent_for_assignment = \
            mock.MagicMock(return_value=self.agent_1)

        onboard_threads = manager.assignment_to_onboard_thread
        did_launch = manager._onboard_new_agent(self.agent_1)
        assert_equal_by(
            onboard_threads[self.agent_1.assignment_id].isAlive, True, 0.2)
        time.sleep(0.1)
        manager.worker_manager.change_agent_conversation.assert_called_once()
        manager.worker_manager.change_agent_conversation.reset_mock()
        manager.onboard_function.assert_not_called()
        self.assertTrue(did_launch)

        # Thread will be waiting for agent_1 status to go to ONBOARDING, ensure
        # won't start new thread on a repeat call when first still alive
        did_launch = manager._onboard_new_agent(self.agent_1)
        time.sleep(0.2)
        manager.worker_manager.change_agent_conversation.assert_not_called()
        manager.worker_manager.get_agent_for_assignment.assert_not_called()
        manager.onboard_function.assert_not_called()
        self.assertFalse(did_launch)

        # Advance the worker to simulate a connection, assert onboarding goes
        self.agent_1.set_status(AssignState.STATUS_ONBOARDING)
        assert_equal_by(
            onboard_threads[self.agent_1.assignment_id].isAlive, False, 0.6)
        manager.onboard_function.assert_called_with(self.agent_1)
        manager._move_agents_to_waiting.assert_called_once()

        # Try to launch a new onboarding world for the same agent still in
        # onboarding, assert that this call is ignored.
        did_launch = manager._onboard_new_agent(self.agent_1)
        self.assertFalse(did_launch)

        # Try to launch with an agent that was in none but supposedly launched
        # before
        self.agent_1.set_status(AssignState.STATUS_NONE)
        did_launch = manager._onboard_new_agent(self.agent_1)
        self.assertTrue(did_launch)
        self.agent_1.set_status(AssignState.STATUS_ONBOARDING)

    def test_agents_incomplete(self):
        agents = [self.agent_1, self.agent_2, self.agent_3]
        manager = self.mturk_manager
        self.assertFalse(manager._no_agents_incomplete(agents))
        self.agent_1.set_status(AssignState.STATUS_DISCONNECT)
        self.assertFalse(manager._no_agents_incomplete(agents))
        self.agent_2.set_status(AssignState.STATUS_DONE)
        self.assertFalse(manager._no_agents_incomplete(agents))
        self.agent_3.set_status(AssignState.STATUS_PARTNER_DISCONNECT)
        self.assertFalse(manager._no_agents_incomplete(agents))
        self.agent_1.set_status(AssignState.STATUS_DONE)
        self.assertFalse(manager._no_agents_incomplete(agents))
        self.agent_3.set_status(AssignState.STATUS_DONE)
        self.assertTrue(manager._no_agents_incomplete(agents))

    def test_world_types(self):
        onboard_type = 'o_12345'
        waiting_type = 'w_12345'
        task_type = 't_12345'
        garbage_type = 'g_12345'
        manager = self.mturk_manager
        self.assertTrue(manager.is_onboarding_world(onboard_type))
        self.assertTrue(manager.is_task_world(task_type))
        self.assertTrue(manager.is_waiting_world(waiting_type))
        for world_type in [waiting_type, task_type, garbage_type]:
            self.assertFalse(manager.is_onboarding_world(world_type))
        for world_type in [onboard_type, task_type, garbage_type]:
            self.assertFalse(manager.is_waiting_world(world_type))
        for world_type in [waiting_type, onboard_type, garbage_type]:
            self.assertFalse(manager.is_task_world(world_type))

    def test_turk_timeout(self):
        '''Timeout should send expiration message to worker and be treated as
        a disconnect event.'''
        manager = self.mturk_manager
        manager.force_expire_hit = mock.MagicMock()
        manager._handle_agent_disconnect = mock.MagicMock()

        manager.handle_turker_timeout(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        manager.force_expire_hit.assert_called_once()
        call_args = manager.force_expire_hit.call_args
        self.assertEqual(call_args[0][0], TEST_WORKER_ID_1)
        self.assertEqual(call_args[0][1], TEST_ASSIGNMENT_ID_1)
        manager._handle_agent_disconnect.assert_called_once_with(
            TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1
        )

    def test_wait_for_task_expirations(self):
        '''Ensure waiting for expiration time actually works out'''
        manager = self.mturk_manager
        manager.opt['assignment_duration_in_seconds'] = 0.5
        manager.expire_all_unassigned_hits = mock.MagicMock()
        manager.hit_id_list = [1, 2, 3]

        def run_task_wait():
            manager._wait_for_task_expirations()

        wait_thread = threading.Thread(target=run_task_wait, daemon=True)
        wait_thread.start()
        time.sleep(0.1)
        self.assertTrue(wait_thread.isAlive())
        assert_equal_by(wait_thread.isAlive, False, 0.6)

    def test_mark_workers_done(self):
        manager = self.mturk_manager
        manager.give_worker_qualification = mock.MagicMock()
        manager._log_working_time = mock.MagicMock()
        manager.has_time_limit = False

        # Assert finality doesn't change
        self.agent_1.set_status(AssignState.STATUS_DISCONNECT)
        manager.mark_workers_done([self.agent_1])
        self.assertEqual(
            AssignState.STATUS_DISCONNECT, self.agent_1.get_status())

        # assert uniqueness works as expected
        manager.is_unique = True
        with self.assertRaises(AssertionError):
            manager.mark_workers_done([self.agent_2])
        manager.give_worker_qualification.assert_not_called()
        manager.unique_qual_name = 'fake_qual_name'
        manager.mark_workers_done([self.agent_2])
        manager.give_worker_qualification.assert_called_once_with(
            self.agent_2.worker_id, 'fake_qual_name')
        self.assertEqual(self.agent_2.get_status(), AssignState.STATUS_DONE)
        manager.is_unique = False

        # Ensure working time is called if it's set
        manager.has_time_limit = True
        manager.mark_workers_done([self.agent_3])
        self.assertEqual(self.agent_3.get_status(), AssignState.STATUS_DONE)
        manager._log_working_time.assert_called_once_with(self.agent_3)


class TestMTurkManagerPoolHandling(unittest.TestCase):
    def setUp(self):
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args(print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(
            opt=self.opt,
            mturk_agent_ids=self.mturk_agent_ids,
            is_test=True,
        )
        self.mturk_manager._init_state()
        self.agent_1 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_1,
                                  TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1)
        self.agent_2 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_2,
                                  TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2)
        self.agent_3 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_3,
                                  TEST_ASSIGNMENT_ID_3, TEST_WORKER_ID_3)

    def tearDown(self):
        self.mturk_manager.shutdown()

    def test_pool_add_get_remove_and_expire(self):
        '''Ensure the pool properly adds and releases workers'''
        all_are_eligible = {
            'multiple': True,
            'func': lambda workers: workers,
        }
        manager = self.mturk_manager

        # Test empty pool
        pool = manager._get_unique_pool(all_are_eligible)
        self.assertEqual(pool, [])

        # Test pool add and get
        manager._add_agent_to_pool(self.agent_1)
        manager._add_agent_to_pool(self.agent_2)
        manager._add_agent_to_pool(self.agent_3)
        self.assertListEqual(manager._get_unique_pool(all_are_eligible), [
            self.agent_1, self.agent_2, self.agent_3])

        # Test extra add to pool has no effect
        manager._add_agent_to_pool(self.agent_1)
        self.assertListEqual(manager._get_unique_pool(all_are_eligible), [
            self.agent_1, self.agent_2, self.agent_3])

        # Test remove from the pool works:
        manager._remove_from_agent_pool(self.agent_2)
        self.assertListEqual(manager._get_unique_pool(all_are_eligible), [
            self.agent_1, self.agent_3])

        # Test repeated remove fails
        with self.assertRaises(AssertionError):
            manager._remove_from_agent_pool(self.agent_2)

        # Test eligibility function
        second_worker_only = {
            'multiple': True,
            'func': lambda workers: [workers[1]],
        }
        self.assertListEqual(manager._get_unique_pool(second_worker_only), [
            self.agent_3])

        # Test single eligibility function
        only_agent_1 = {
            'multiple': False,
            'func': lambda worker: worker is self.agent_1,
        }
        self.assertListEqual(manager._get_unique_pool(only_agent_1), [
            self.agent_1])

        # Test expiration of pool
        manager.force_expire_hit = mock.MagicMock()

        manager._expire_agent_pool()
        manager.force_expire_hit.assert_any_call(self.agent_1.worker_id,
                                                 self.agent_1.assignment_id)
        manager.force_expire_hit.assert_any_call(self.agent_3.worker_id,
                                                 self.agent_3.assignment_id)
        pool = manager._get_unique_pool(all_are_eligible)
        self.assertEqual(pool, [])

        # Test adding two agents from the same worker
        self.agent_2.worker_id = self.agent_1.worker_id
        manager._add_agent_to_pool(self.agent_1)
        manager._add_agent_to_pool(self.agent_2)
        # both workers are in the pool
        self.assertListEqual(manager.agent_pool, [self.agent_1, self.agent_2])
        # Only one worker per unique list though
        manager.is_sandbox = False
        self.assertListEqual(manager._get_unique_pool(all_are_eligible), [
            self.agent_1])


class TestMTurkManagerTimeHandling(unittest.TestCase):
    def setUp(self):
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args(print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(
            opt=self.opt,
            mturk_agent_ids=self.mturk_agent_ids,
            is_test=True,
        )
        self.mturk_manager.time_limit_checked = time.time()
        self.mturk_manager.worker_manager.un_time_block_workers = \
            mock.MagicMock()
        self.mturk_manager.worker_manager.time_block_worker = mock.MagicMock()
        self.old_time = MTurkManagerFile.time
        MTurkManagerFile.time = mock.MagicMock()
        MTurkManagerFile.time.time = mock.MagicMock(return_value=0)
        self.agent_1 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_1,
                                  TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1)
        self.agent_2 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_2,
                                  TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2)

    def tearDown(self):
        self.mturk_manager.shutdown()
        MTurkManagerFile.time = self.old_time

    def test_create_work_time_file(self):
        manager = self.mturk_manager
        manager._should_use_time_logs = mock.MagicMock(return_value=True)

        file_path = os.path.join(parent_dir,
                                 MTurkManagerFile.TIME_LOGS_FILE_NAME)
        file_lock = os.path.join(parent_dir,
                                 MTurkManagerFile.TIME_LOGS_FILE_LOCK)
        # No lock should exist already
        self.assertFalse(os.path.exists(file_lock))

        # open the work time file, ensure it was just updated
        MTurkManagerFile.time.time = mock.MagicMock(return_value=42424242)
        manager._reset_time_logs(force=True)
        with open(file_path, 'rb+') as time_log_file:
            existing_times = pickle.load(time_log_file)
            self.assertEqual(existing_times['last_reset'], 42424242)
            self.assertEqual(len(existing_times), 1)

        # Try to induce a check, ensure it doesn't fire because too recent
        MTurkManagerFile.time.time = \
            mock.MagicMock(return_value=(60 * 60 * 24 * 1000))
        manager._check_time_limit()
        manager.worker_manager.un_time_block_workers.assert_not_called()

        # Try to induce a check, ensure it doesn't fire because outside of 30
        # minute window
        MTurkManagerFile.time.time = mock.MagicMock(
            return_value=(60 * 60 * 24 * 1000) + (60 * 40))
        manager.time_limit_checked = 0
        manager._check_time_limit()
        manager.worker_manager.un_time_block_workers.assert_not_called()

        # Induce a check
        MTurkManagerFile.time.time = \
            mock.MagicMock(return_value=(60 * 60 * 24 * 1000))
        manager._check_time_limit()
        self.assertEqual(manager.time_limit_checked, (60 * 60 * 24 * 1000))

    def test_add_to_work_time_file_and_block(self):
        manager = self.mturk_manager
        self.agent_1.creation_time = 1000
        self.agent_2.creation_time = 1000
        manager.opt['max_time'] = 10000
        # Ensure a worker below the time limit isn't blocked
        MTurkManagerFile.time.time = mock.MagicMock(return_value=10000)
        self.mturk_manager._should_use_time_logs = \
            mock.MagicMock(return_value=True)
        manager._log_working_time(self.agent_1)
        manager.worker_manager.time_block_worker.assert_not_called()

        # Ensure a worker above the time limit is blocked
        MTurkManagerFile.time.time = mock.MagicMock(return_value=100000)
        manager._log_working_time(self.agent_2)
        manager.worker_manager.time_block_worker.assert_called_with(
            self.agent_2.worker_id)

        # Ensure on a (forced) reset all workers are freed
        manager._reset_time_logs(force=True)
        manager.worker_manager.un_time_block_workers.assert_called_once()
        args = manager.worker_manager.un_time_block_workers.call_args
        worker_list = args[0][0]
        self.assertIn(self.agent_1.worker_id, worker_list)
        self.assertIn(self.agent_2.worker_id, worker_list)


class TestMTurkManagerLifecycleFunctions(unittest.TestCase):
    def setUp(self):
        self.fake_socket = MockSocket()
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args(print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['task_description'] = 'Test task description'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(
            opt=self.opt,
            mturk_agent_ids=self.mturk_agent_ids,
            is_test=True,
        )
        MTurkManagerFile.server_utils.delete_server = mock.MagicMock()

    def tearDown(self):
        self.mturk_manager.shutdown()
        self.fake_socket.close()

    def test_full_lifecycle(self):
        manager = self.mturk_manager
        server_url = 'https://fake_server_url'
        topic_arn = 'aws_topic_arn'
        mturk_page_url = 'https://test_mturk_page_url'
        MTurkManagerFile.server_utils.setup_server = \
            mock.MagicMock(return_value=server_url)
        MTurkManagerFile.server_utils.setup_legacy_server = \
            mock.MagicMock(return_value=server_url)

        # Currently in state created. Try steps that are too soon to work
        with self.assertRaises(AssertionError):
            manager.start_new_run()
        with self.assertRaises(AssertionError):
            manager.start_task(None, None, None)

        # Setup the server but fail due to insufficent funds
        manager.opt['local'] = True
        MTurkManagerFile.input = mock.MagicMock()
        MTurkManagerFile.mturk_utils.setup_aws_credentials = mock.MagicMock()
        MTurkManagerFile.mturk_utils.check_mturk_balance = \
            mock.MagicMock(return_value=False)
        MTurkManagerFile.mturk_utils.calculate_mturk_cost = \
            mock.MagicMock(return_value=10)
        with self.assertRaises(SystemExit):
            manager.setup_server()

        MTurkManagerFile.mturk_utils.setup_aws_credentials.assert_called_once()
        MTurkManagerFile.mturk_utils.check_mturk_balance.assert_called_once()
        MTurkManagerFile.input.assert_called()
        # Two calls to to input if local is set
        self.assertEqual(len(MTurkManagerFile.input.call_args_list), 2)

        # Test successful setup
        manager.opt['local'] = False
        MTurkManagerFile.input.reset_mock()
        MTurkManagerFile.mturk_utils.check_mturk_balance = \
            mock.MagicMock(return_value=True)
        MTurkManagerFile.mturk_utils.create_hit_config = mock.MagicMock()
        manager.setup_server()
        # Copy one file for cover page, 2 workers, and 1 onboarding
        self.assertEqual(len(manager.task_files_to_copy), 4)
        self.assertEqual(manager.server_url, server_url)
        self.assertIn('unittest', manager.server_task_name)
        MTurkManagerFile.input.assert_called_once()
        MTurkManagerFile.mturk_utils.check_mturk_balance.assert_called_once()
        MTurkManagerFile.mturk_utils.create_hit_config.assert_called_once()
        self.assertEqual(manager.task_state, manager.STATE_SERVER_ALIVE)

        # Start a new run
        MTurkManagerFile.mturk_utils.setup_sns_topic = \
            mock.MagicMock(return_value=topic_arn)
        manager._init_state = mock.MagicMock(wraps=manager._init_state)
        manager.start_new_run()
        manager._init_state.assert_called_once()
        MTurkManagerFile.mturk_utils.setup_sns_topic.assert_called_once_with(
            manager.opt['task'], manager.server_url, manager.task_group_id,
        )
        self.assertEqual(manager.topic_arn, topic_arn)
        self.assertEqual(manager.task_state, manager.STATE_INIT_RUN)

        # connect to the server
        manager._setup_socket = mock.MagicMock()
        manager.ready_to_accept_workers()
        manager._setup_socket.assert_called_once()
        self.assertEqual(manager.task_state,
                         MTurkManager.STATE_ACCEPTING_WORKERS)

        # 'launch' some hits
        manager.create_additional_hits = \
            mock.MagicMock(return_value=mturk_page_url)
        hits_url = manager.create_hits()
        manager.create_additional_hits.assert_called_once()
        self.assertEqual(manager.task_state, MTurkManager.STATE_HITS_MADE)
        self.assertEqual(hits_url, mturk_page_url)

        # start a task
        manager.num_conversations = 10
        manager.expire_all_unassigned_hits = mock.MagicMock()
        manager._expire_onboarding_pool = mock.MagicMock()
        manager._expire_agent_pool = mock.MagicMock()

        # Run a task, ensure it closes when the max convs have been 'had'
        def run_task():
            manager.start_task(lambda worker: True, None, None)

        task_thread = threading.Thread(target=run_task, daemon=True)
        task_thread.start()

        self.assertTrue(task_thread.isAlive())
        manager.started_conversations = 10
        manager.completed_conversations = 10
        assert_equal_by(task_thread.isAlive, False, 0.6)
        manager.expire_all_unassigned_hits.assert_called_once()
        manager._expire_onboarding_pool.assert_called_once()
        manager._expire_agent_pool.assert_called_once()

        # shutdown
        manager.expire_all_unassigned_hits = mock.MagicMock()
        manager._expire_onboarding_pool = mock.MagicMock()
        manager._expire_agent_pool = mock.MagicMock()
        manager._wait_for_task_expirations = mock.MagicMock()
        MTurkManagerFile.mturk_utils.delete_sns_topic = mock.MagicMock()
        manager.shutdown()
        self.assertTrue(manager.is_shutdown)
        manager.expire_all_unassigned_hits.assert_called_once()
        manager._expire_onboarding_pool.assert_called_once()
        manager._expire_agent_pool.assert_called_once()
        manager._wait_for_task_expirations.assert_called_once()
        MTurkManagerFile.server_utils.delete_server.assert_called_once()
        MTurkManagerFile.mturk_utils.delete_sns_topic.assert_called_once_with(
            topic_arn)


class TestMTurkManagerConnectedFunctions(unittest.TestCase):
    '''Semi-unit semi-integration tests on the more state-dependent
    MTurkManager functionality'''
    def setUp(self):
        self.fake_socket = MockSocket()
        time.sleep(0.1)
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args(print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(
            opt=self.opt,
            mturk_agent_ids=self.mturk_agent_ids,
            is_test=True,
        )
        self.mturk_manager._init_state()
        self.mturk_manager.port = self.fake_socket.port
        self.mturk_manager._onboard_new_agent = mock.MagicMock()
        self.mturk_manager._wait_for_task_expirations = mock.MagicMock()
        self.mturk_manager.task_group_id = 'TEST_GROUP_ID'
        self.mturk_manager.server_url = 'https://127.0.0.1'
        self.mturk_manager.task_state = \
            self.mturk_manager.STATE_ACCEPTING_WORKERS
        self.mturk_manager._setup_socket()
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_1,
            'hit_id': TEST_HIT_ID_1,
            'assignment_id': TEST_ASSIGNMENT_ID_1,
            'conversation_id': None,
        }, '')
        self.mturk_manager._on_alive(alive_packet)
        alive_packet = Packet('', '', '', '', '', {
            'worker_id': TEST_WORKER_ID_2,
            'hit_id': TEST_HIT_ID_2,
            'assignment_id': TEST_ASSIGNMENT_ID_2,
            'conversation_id': None,
        }, '')
        self.mturk_manager._on_alive(alive_packet)
        self.agent_1 = \
            self.mturk_manager.worker_manager.get_agent_for_assignment(
                TEST_ASSIGNMENT_ID_1)
        self.agent_2 = \
            self.mturk_manager.worker_manager.get_agent_for_assignment(
                TEST_ASSIGNMENT_ID_2)

    def tearDown(self):
        self.mturk_manager.shutdown()
        self.fake_socket.close()

    def test_socket_dead(self):
        '''Test all states of socket dead calls'''
        manager = self.mturk_manager
        agent = self.agent_1
        worker_id = agent.worker_id
        assignment_id = agent.assignment_id
        manager.socket_manager.close_channel = mock.MagicMock()
        agent.reduce_state = mock.MagicMock()
        agent.set_status = mock.MagicMock(wraps=agent.set_status)
        manager._handle_agent_disconnect = \
            mock.MagicMock(wraps=manager._handle_agent_disconnect)

        # Test status none
        agent.set_status(AssignState.STATUS_NONE)
        agent.set_status.reset_mock()
        manager._on_socket_dead(worker_id, assignment_id)
        self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
        agent.reduce_state.assert_called_once()
        manager.socket_manager.close_channel.assert_called_once_with(
            agent.get_connection_id()
        )
        manager._handle_agent_disconnect.assert_not_called()

        # Test status onboarding
        agent.set_status(AssignState.STATUS_ONBOARDING)
        agent.set_status.reset_mock()
        agent.reduce_state.reset_mock()
        manager.socket_manager.close_channel.reset_mock()
        self.assertFalse(agent.disconnected)
        manager._on_socket_dead(worker_id, assignment_id)
        self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
        agent.reduce_state.assert_called_once()
        manager.socket_manager.close_channel.assert_called_once_with(
            agent.get_connection_id()
        )
        self.assertTrue(agent.disconnected)
        manager._handle_agent_disconnect.assert_not_called()

        # test status waiting
        agent.disconnected = False
        agent.set_status(AssignState.STATUS_WAITING)
        agent.set_status.reset_mock()
        agent.reduce_state.reset_mock()
        manager.socket_manager.close_channel.reset_mock()
        manager._add_agent_to_pool(agent)
        manager._remove_from_agent_pool = mock.MagicMock()
        manager._on_socket_dead(worker_id, assignment_id)
        self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
        agent.reduce_state.assert_called_once()
        manager.socket_manager.close_channel.assert_called_once_with(
            agent.get_connection_id()
        )
        self.assertTrue(agent.disconnected)
        manager._handle_agent_disconnect.assert_not_called()
        manager._remove_from_agent_pool.assert_called_once_with(agent)

        # test status in task
        agent.disconnected = False
        agent.set_status(AssignState.STATUS_IN_TASK)
        agent.set_status.reset_mock()
        agent.reduce_state.reset_mock()
        manager.socket_manager.close_channel.reset_mock()
        manager._add_agent_to_pool(agent)
        manager._remove_from_agent_pool = mock.MagicMock()
        manager._on_socket_dead(worker_id, assignment_id)
        self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
        manager.socket_manager.close_channel.assert_called_once_with(
            agent.get_connection_id()
        )
        self.assertTrue(agent.disconnected)
        manager._handle_agent_disconnect.assert_called_once_with(
            worker_id, assignment_id)

        # test status done
        agent.disconnected = False
        agent.set_status(AssignState.STATUS_DONE)
        agent.set_status.reset_mock()
        agent.reduce_state.reset_mock()
        manager._handle_agent_disconnect.reset_mock()
        manager.socket_manager.close_channel.reset_mock()
        manager._add_agent_to_pool(agent)
        manager._remove_from_agent_pool = mock.MagicMock()
        manager._on_socket_dead(worker_id, assignment_id)
        self.assertNotEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
        agent.reduce_state.assert_not_called()
        manager.socket_manager.close_channel.assert_not_called()
        self.assertFalse(agent.disconnected)
        manager._handle_agent_disconnect.assert_not_called()

    def test_send_message_command(self):
        manager = self.mturk_manager
        agent = self.agent_1
        worker_id = self.agent_1.worker_id
        assignment_id = self.agent_1.assignment_id
        agent.set_last_command = mock.MagicMock()
        manager.socket_manager.queue_packet = mock.MagicMock()

        # Send a command
        data = {'text': data_model.COMMAND_SEND_MESSAGE}
        manager.send_command(worker_id, assignment_id, data)

        agent.set_last_command.assert_called_once_with(data)
        manager.socket_manager.queue_packet.assert_called_once()
        packet = manager.socket_manager.queue_packet.call_args[0][0]
        self.assertIsNotNone(packet.id)
        self.assertEqual(packet.type, Packet.TYPE_MESSAGE)
        self.assertEqual(packet.receiver_id, worker_id)
        self.assertEqual(packet.assignment_id, assignment_id)
        self.assertEqual(packet.data, data)
        self.assertEqual(packet.data['type'], data_model.MESSAGE_TYPE_COMMAND)

        # Send a message
        data = {'text': 'This is a test message'}
        agent.set_last_command.reset_mock()
        manager.socket_manager.queue_packet.reset_mock()
        message_id = manager.send_message(worker_id, assignment_id, data)
        agent.set_last_command.assert_not_called()
        manager.socket_manager.queue_packet.assert_called_once()
        packet = manager.socket_manager.queue_packet.call_args[0][0]
        self.assertIsNotNone(packet.id)
        self.assertEqual(packet.type, Packet.TYPE_MESSAGE)
        self.assertEqual(packet.receiver_id, worker_id)
        self.assertEqual(packet.assignment_id, assignment_id)
        self.assertNotEqual(packet.data, data)
        self.assertEqual(data['text'], packet.data['text'])
        self.assertEqual(packet.data['message_id'], message_id)
        self.assertEqual(packet.data['type'], data_model.MESSAGE_TYPE_MESSAGE)

    def test_free_workers(self):
        manager = self.mturk_manager
        manager.socket_manager.close_channel = mock.MagicMock()
        manager.free_workers([self.agent_1])
        manager.socket_manager.close_channel.assert_called_once_with(
            self.agent_1.get_connection_id())

    def test_force_expire_hit(self):
        manager = self.mturk_manager
        agent = self.agent_1
        worker_id = agent.worker_id
        assignment_id = agent.assignment_id
        socket_manager = manager.socket_manager
        manager.send_command = mock.MagicMock()
        socket_manager.close_channel = mock.MagicMock()

        # Test expiring finished worker
        agent.set_status(AssignState.STATUS_DONE)
        manager.force_expire_hit(worker_id, assignment_id)
        manager.send_command.assert_not_called()
        socket_manager.close_channel.assert_not_called()
        self.assertEqual(agent.get_status(), AssignState.STATUS_DONE)

        # Test expiring not finished worker with default args
        agent.set_status(AssignState.STATUS_ONBOARDING)
        manager.force_expire_hit(worker_id, assignment_id)
        manager.send_command.assert_called_once()
        args = manager.send_command.call_args[0]
        used_worker_id, used_assignment_id, data = args[0], args[1], args[2]
        ack_func = manager.send_command.call_args[1]['ack_func']
        ack_func()
        self.assertEqual(worker_id, used_worker_id)
        self.assertEqual(assignment_id, used_assignment_id)
        self.assertEqual(data['text'], data_model.COMMAND_EXPIRE_HIT)
        self.assertEqual(agent.get_status(), AssignState.STATUS_EXPIRED)
        self.assertTrue(agent.hit_is_expired)
        self.assertIsNotNone(data['inactive_text'])
        socket_manager.close_channel.assert_called_once_with(
            agent.get_connection_id())

        # Test expiring not finished worker with custom arguments
        agent.set_status(AssignState.STATUS_ONBOARDING)
        agent.hit_is_expired = False
        manager.send_command = mock.MagicMock()
        socket_manager.close_channel = mock.MagicMock()
        special_disconnect_text = 'You were disconnected as part of a test'
        test_ack_function = mock.MagicMock()
        manager.force_expire_hit(
            worker_id, assignment_id,
            text=special_disconnect_text, ack_func=test_ack_function)
        manager.send_command.assert_called_once()
        args = manager.send_command.call_args[0]
        used_worker_id, used_assignment_id, data = args[0], args[1], args[2]
        ack_func = manager.send_command.call_args[1]['ack_func']
        ack_func()
        self.assertEqual(worker_id, used_worker_id)
        self.assertEqual(assignment_id, used_assignment_id)
        self.assertEqual(data['text'], data_model.COMMAND_EXPIRE_HIT)
        self.assertEqual(agent.get_status(), AssignState.STATUS_EXPIRED)
        self.assertTrue(agent.hit_is_expired)
        self.assertEqual(data['inactive_text'], special_disconnect_text)
        socket_manager.close_channel.assert_called_once_with(
            agent.get_connection_id())
        test_ack_function.assert_called()

    def test_get_qualifications(self):
        manager = self.mturk_manager
        mturk_utils = MTurkManagerFile.mturk_utils
        mturk_utils.find_or_create_qualification = mock.MagicMock()

        # create a qualification list with nothing but a provided junk qual
        fake_qual = {
            'QualificationTypeId': 'fake_qual_id',
            'Comparator': 'DoesNotExist',
            'ActionsGuarded': 'DiscoverPreviewAndAccept'
        }
        qualifications = manager.get_qualification_list([fake_qual])
        self.assertListEqual(qualifications, [fake_qual])
        self.assertListEqual(manager.qualifications, [fake_qual])
        mturk_utils.find_or_create_qualification.assert_not_called()

        # Create a qualificaiton list using all the default types
        disconnect_qual_name = 'disconnect_qual_name'
        disconnect_qual_id = 'disconnect_qual_id'
        block_qual_name = 'block_qual_name'
        block_qual_id = 'block_qual_id'
        max_time_qual_name = 'max_time_qual_name'
        max_time_qual_id = 'max_time_qual_id'
        unique_qual_name = 'unique_qual_name'
        unique_qual_id = 'unique_qual_id'

        def return_qualifications(qual_name, _text, _sb):
            if qual_name == disconnect_qual_name:
                return disconnect_qual_id
            if qual_name == block_qual_name:
                return block_qual_id
            if qual_name == max_time_qual_name:
                return max_time_qual_id
            if qual_name == unique_qual_name:
                return unique_qual_id

        mturk_utils.find_or_create_qualification = return_qualifications
        manager.opt['disconnect_qualification'] = disconnect_qual_name
        manager.opt['block_qualification'] = block_qual_name
        manager.opt['max_time_qual'] = max_time_qual_name
        manager.opt['unique_qual_name'] = unique_qual_name
        manager.is_unique = True
        manager.has_time_limit = True
        manager.qualifications = None
        qualifications = manager.get_qualification_list()

        for qual in qualifications:
            self.assertEqual(qual['ActionsGuarded'],
                             'DiscoverPreviewAndAccept')
            self.assertEqual(qual['Comparator'], 'DoesNotExist')

        for qual_id in [disconnect_qual_id, block_qual_id,
                        max_time_qual_id, unique_qual_id]:
            has_qual = False
            for qual in qualifications:
                if qual['QualificationTypeId'] == qual_id:
                    has_qual = True
                    break
            self.assertTrue(has_qual)

        self.assertListEqual(qualifications, manager.qualifications)

    def test_create_additional_hits(self):
        manager = self.mturk_manager
        manager.opt['hit_title'] = 'test_hit_title'
        manager.opt['hit_description'] = 'test_hit_description'
        manager.opt['hit_keywords'] = 'test_hit_keywords'
        manager.opt['reward'] = 0.1
        mturk_utils = MTurkManagerFile.mturk_utils
        fake_hit = 'fake_hit_type'
        mturk_utils.create_hit_type = mock.MagicMock(return_value=fake_hit)
        mturk_utils.subscribe_to_hits = mock.MagicMock()
        mturk_utils.create_hit_with_hit_type = mock.MagicMock(
            return_value=('page_url', 'hit_id', 'test_hit_response')
        )
        manager.server_url = 'test_url'
        manager.task_group_id = 'task_group_id'
        manager.topic_arn = 'topic_arn'
        mturk_chat_url = '{}/chat_index?task_group_id={}'.format(
            manager.server_url,
            manager.task_group_id
        )
        hit_url = manager.create_additional_hits(5)
        mturk_utils.create_hit_type.assert_called_once()
        mturk_utils.subscribe_to_hits.assert_called_with(
            fake_hit, manager.is_sandbox, manager.topic_arn)
        self.assertEqual(
            len(mturk_utils.create_hit_with_hit_type.call_args_list), 5)
        mturk_utils.create_hit_with_hit_type.assert_called_with(
            opt=manager.opt,
            page_url=mturk_chat_url,
            hit_type_id=fake_hit,
            num_assignments=1,
            is_sandbox=manager.is_sandbox,
        )
        self.assertEqual(len(manager.hit_id_list), 5)
        self.assertEqual(hit_url, 'page_url')

    def test_expire_all_hits(self):
        manager = self.mturk_manager
        worker_manager = manager.worker_manager
        completed_hit_id = 'completed'
        incomplete_1 = 'incomplete_1'
        incomplete_2 = 'incomplete_2'
        MTurkManagerFile.mturk_utils.expire_hit = mock.MagicMock()
        worker_manager.get_complete_hits = \
            mock.MagicMock(return_value=[completed_hit_id])
        manager.hit_id_list = [completed_hit_id, incomplete_1, incomplete_2]

        manager.expire_all_unassigned_hits()
        worker_manager.get_complete_hits.assert_called_once()
        expire_calls = MTurkManagerFile.mturk_utils.expire_hit.call_args_list
        self.assertEqual(len(expire_calls), 2)
        for hit in [incomplete_1, incomplete_2]:
            found = False
            for expire_call in expire_calls:
                if expire_call[0][1] == hit:
                    found = True
                    break
            self.assertTrue(found)

    def test_qualification_management(self):
        manager = self.mturk_manager
        test_qual_name = 'test_qual'
        other_qual_name = 'other_qual'
        test_qual_id = 'test_qual_id'
        worker_id = self.agent_1.worker_id
        mturk_utils = MTurkManagerFile.mturk_utils
        success_id = 'Success'

        def find_qualification(qual_name, _sandbox):
            if qual_name == test_qual_name:
                return test_qual_id
            return None

        mturk_utils.find_qualification = find_qualification
        mturk_utils.give_worker_qualification = mock.MagicMock()
        mturk_utils.remove_worker_qualification = mock.MagicMock()
        mturk_utils.find_or_create_qualification = \
            mock.MagicMock(return_value=success_id)

        # Test give qualification
        manager.give_worker_qualification(worker_id, test_qual_name)
        mturk_utils.give_worker_qualification.assert_called_once_with(
            worker_id, test_qual_id, None, manager.is_sandbox
        )

        # Test revoke qualification
        manager.remove_worker_qualification(worker_id, test_qual_name)
        mturk_utils.remove_worker_qualification.assert_called_once_with(
            worker_id, test_qual_id, manager.is_sandbox, ''
        )

        # Test create qualification can exist
        result = manager.create_qualification(test_qual_name, '')
        self.assertEqual(result, success_id)

        # Test create qualification can't exist failure
        result = manager.create_qualification(test_qual_name, '', False)
        self.assertIsNone(result)

        # Test create qualification can't exist success
        result = manager.create_qualification(other_qual_name, '')
        self.assertEqual(result, success_id)

    def test_partner_disconnect(self):
        manager = self.mturk_manager
        manager.send_command = mock.MagicMock()
        self.agent_1.set_status(AssignState.STATUS_IN_TASK)
        manager._handle_partner_disconnect(self.agent_1)
        self.assertEqual(
            self.agent_1.get_status(), AssignState.STATUS_PARTNER_DISCONNECT)
        args = manager.send_command.call_args[0]
        worker_id, assignment_id, data = args[0], args[1], args[2]
        self.assertEqual(worker_id, self.agent_1.worker_id)
        self.assertEqual(assignment_id, self.agent_1.assignment_id)
        self.assertDictEqual(data, self.agent_1.get_inactive_command_data())

    def test_restore_state(self):
        manager = self.mturk_manager
        worker_manager = manager.worker_manager
        worker_manager.change_agent_conversation = mock.MagicMock()
        manager.send_command = mock.MagicMock()

        agent = self.agent_1
        agent.conversation_id = 'Test_conv_id'
        agent.id = 'test_agent_id'
        agent.request_message = mock.MagicMock()
        agent.message_request_time = time.time()
        test_message = {'text': 'this_is_a_message', 'message_id': 'test_id',
                        'type': data_model.MESSAGE_TYPE_MESSAGE}
        agent.append_message(test_message)
        manager._restore_agent_state(agent.worker_id, agent.assignment_id)
        self.assertFalse(agent.alived)
        manager.send_command.assert_not_called()
        worker_manager.change_agent_conversation.assert_called_once_with(
            agent=agent, conversation_id=agent.conversation_id,
            new_agent_id=agent.id,
        )
        agent.alived = True
        assert_equal_by(
            lambda: len(agent.request_message.call_args_list), 1, 0.6)
        manager.send_command.assert_called_once()
        args = manager.send_command.call_args[0]
        worker_id, assignment_id, data = args[0], args[1], args[2]
        self.assertEqual(worker_id, agent.worker_id)
        self.assertEqual(assignment_id, agent.assignment_id)
        self.assertListEqual(data['messages'], agent.get_messages())
        self.assertEqual(data['text'], data_model.COMMAND_RESTORE_STATE)

    def test_expire_onboarding(self):
        manager = self.mturk_manager
        manager.force_expire_hit = mock.MagicMock()
        self.agent_2.set_status(AssignState.STATUS_ONBOARDING)
        manager._expire_onboarding_pool()

        manager.force_expire_hit.assert_called_once_with(
            self.agent_2.worker_id, self.agent_2.assignment_id,
        )


if __name__ == '__main__':
    unittest.main(buffer=True)
