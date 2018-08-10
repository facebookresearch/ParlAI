import unittest
import os
import time
import json
import threading
from unittest import mock
from parlai.mturk.core.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.agents import MTurkAgent, AssignState
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.core.socket_manager import SocketManager, Packet
from parlai.core.params import ParlaiParser
from websocket_server import WebsocketServer

import parlai.mturk.core.mturk_manager as MTurkManagerFile
import parlai.mturk.core.data_model as data_model

parent_dir = os.path.dirname(os.path.abspath(__file__))
MTurkManagerFile.parent_dir = os.path.dirname(os.path.abspath(__file__))

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


class MockSocket():
    def __init__(self):
        self.last_messages = {}
        self.connected = False
        self.disconnected = False
        self.closed = False
        self.ws = None
        self.should_heartbeat = True
        self.fake_workers = []
        self.launch_socket()
        self.handlers = {}

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
                    'type':  data_model.SOCKET_ROUTE_PACKET_STRING,
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
            self.ws = WebsocketServer(3030, host='127.0.0.1')
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
        self.opt = argparser.parse_args()
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(
            opt=self.opt,
            mturk_agent_ids=self.mturk_agent_ids
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
        self.assertEqual(manager.num_conversations,  opt['num_conversations'])
        self.assertEqual(manager.is_sandbox, opt['is_sandbox'])

        self.assertGreaterEqual(
            manager.required_hits,
            manager.num_conversations * len(self.mturk_agent_ids))

        self.assertIsNotNone(manager.agent_pool_change_condition)

        self.assertEqual(manager.minimum_messages, opt.get('min_messages', 0))
        self.assertEqual(manager.auto_approve_delay,
                         opt.get('auto_approve_delay', 4*7*24*3600))
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


class TestMTurkManagerAgentSetup(unittest.TestCase):
    '''Unit tests mturk manager agent setup and handling'''
    def setUp(self):
        self.fake_socket = MockSocket()
        time.sleep(0.1)
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args()
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(
            opt=self.opt,
            mturk_agent_ids=self.mturk_agent_ids
        )
        self.mturk_manager._init_state()
        self.mturk_manager.port = 3030
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
        '''Basic socket setup should fail when not in correct state'''
        with self.assertRaises(AssertionError):
            self.mturk_manager._setup_socket()
        self.mturk_manager.task_group_id = 'TEST_GROUP_ID'
        self.mturk_manager.server_url = 'https://127.0.0.1'
        self.mturk_manager.task_state = \
            self.mturk_manager.STATE_ACCEPTING_WORKERS
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


if __name__ == '__main__':
    unittest.main()
