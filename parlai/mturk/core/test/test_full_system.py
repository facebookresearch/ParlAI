#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import time
import uuid
import os
from unittest import mock
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser

import parlai.mturk.core.mturk_manager as MTurkManagerFile
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json

parent_dir = os.path.dirname(os.path.abspath(__file__))
MTurkManagerFile.parent_dir = os.path.dirname(os.path.abspath(__file__))

# Lets ignore the logging part
MTurkManagerFile.shared_utils.print_and_log = mock.MagicMock()

TEST_WORKER_ID_1 = 'TEST_WORKER_ID_1'
TEST_WORKER_ID_2 = 'TEST_WORKER_ID_2'
TEST_ASSIGNMENT_ID_1 = 'TEST_ASSIGNMENT_ID_1'
TEST_ASSIGNMENT_ID_2 = 'TEST_ASSIGNMENT_ID_2'
TEST_ASSIGNMENT_ID_3 = 'TEST_ASSIGNMENT_ID_3'
TEST_HIT_ID_1 = 'TEST_HIT_ID_1'
TEST_HIT_ID_2 = 'TEST_HIT_ID_2'
TEST_CONV_ID_1 = 'TEST_CONV_ID_1'
FAKE_ID = 'BOGUS'

MESSAGE_ID_1 = 'MESSAGE_ID_1'
MESSAGE_ID_2 = 'MESSAGE_ID_2'
MESSAGE_ID_3 = 'MESSAGE_ID_3'
MESSAGE_ID_4 = 'MESSAGE_ID_4'
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

TASK_GROUP_ID_1 = 'TASK_GROUP_ID_1'

SocketManager.DEF_MISSED_PONGS = 1
SocketManager.HEARTBEAT_RATE = 0.4
SocketManager.DEF_DEAD_TIME = 0.4
SocketManager.ACK_TIME = {Packet.TYPE_ALIVE: 0.4, Packet.TYPE_MESSAGE: 0.2}

shared_utils.THREAD_SHORT_SLEEP = 0.05
shared_utils.THREAD_MEDIUM_SLEEP = 0.15

MTurkManagerFile.WORLD_START_TIMEOUT = 2


TOPIC_ARN = 'topic_arn'
QUALIFICATION_ID = 'qualification_id'
HIT_TYPE_ID = 'hit_type_id'
MTURK_PAGE_URL = 'mturk_page_url'
FAKE_HIT_ID = 'fake_hit_id'


def assert_equal_by(val_func, val, max_time):
    start_time = time.time()
    while val_func() != val:
        assert (
            time.time() - start_time < max_time
        ), "Value was not attained in specified time"
        time.sleep(0.1)


class MockSocket:
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
                self.ws.send_message(client, json.dumps({'type': 'conn_success'}))
                self.connected = True
            elif packet_dict['content']['type'] == 'heartbeat':
                pong = packet_dict['content'].copy()
                pong['type'] = 'pong'
                self.ws.send_message(
                    client,
                    json.dumps(
                        {'type': data_model.SOCKET_ROUTE_PACKET_STRING, 'content': pong}
                    ),
                )
            if 'receiver_id' in packet_dict['content']:
                receiver_id = packet_dict['content']['receiver_id']
                assignment_id = packet_dict['content']['assignment_id']
                use_func = self.handlers.get(
                    receiver_id + assignment_id, self.do_nothing
                )
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
            target=run_socket, name='Fake-Socket-Thread'
        )
        self.listen_thread.daemon = True
        self.listen_thread.start()


class MockAgent(object):
    """
    Class that pretends to be an MTurk agent interacting through the webpage by
    simulating the same commands that are sent from the core.html file.

    Exposes methods to use for testing and checking status
    """

    def __init__(self, hit_id, assignment_id, worker_id, task_group_id):
        self.conversation_id = None
        self.id = None
        self.assignment_id = assignment_id
        self.hit_id = hit_id
        self.worker_id = worker_id
        self.some_agent_disconnected = False
        self.disconnected = False
        self.task_group_id = task_group_id
        self.ws = None
        self.always_beat = True
        self.send_acks = True
        self.ready = False
        self.wants_to_send = False
        self.acked_packet = []
        self.incoming_hb = []
        self.message_packet = []

    def send_packet(self, packet):
        def callback(*args):
            pass

        event_name = data_model.SOCKET_ROUTE_PACKET_STRING
        self.ws.send(json.dumps({'type': event_name, 'content': packet.as_dict()}))

    def register_to_socket(self, ws):
        handler = self.make_packet_handler()
        self.ws = ws
        self.ws.handlers[self.worker_id + self.assignment_id] = handler

    def on_msg(self, packet):
        self.message_packet.append(packet)
        if packet.data['text'] == data_model.COMMAND_CHANGE_CONVERSATION:
            self.ready = False
            self.conversation_id = packet.data['conversation_id']
            self.id = packet.data['agent_id']
            self.send_alive()

    def make_packet_handler(self):
        """
        A packet handler that properly sends heartbeats.
        """

        def on_ack(*args):
            self.acked_packet.append(args[0])

        def on_hb(*args):
            self.incoming_hb.append(args[0])

        def handler_mock(pkt):
            if pkt['type'] == Packet.TYPE_ACK:
                self.ready = True
                packet = Packet.from_dict(pkt)
                on_ack(packet)
            elif pkt['type'] == Packet.TYPE_HEARTBEAT:
                packet = Packet.from_dict(pkt)
                on_hb(packet)
                if self.always_beat:
                    self.send_heartbeat()
            elif pkt['type'] == Packet.TYPE_MESSAGE:
                packet = Packet.from_dict(pkt)
                if self.send_acks:
                    self.send_packet(packet.get_ack())
                self.on_msg(packet)
            elif pkt['type'] == Packet.TYPE_ALIVE:
                raise Exception('Invalid alive packet {}'.format(pkt))
            else:
                raise Exception(
                    'Invalid Packet type {} received in {}'.format(pkt['type'], pkt)
                )

        return handler_mock

    def build_and_send_packet(self, packet_type, data):
        msg = {
            'id': str(uuid.uuid4()),
            'type': packet_type,
            'sender_id': self.worker_id,
            'assignment_id': self.assignment_id,
            'conversation_id': self.conversation_id,
            'receiver_id': '[World_' + self.task_group_id + ']',
            'data': data,
        }

        event_name = data_model.SOCKET_ROUTE_PACKET_STRING
        if packet_type == Packet.TYPE_ALIVE:
            event_name = data_model.SOCKET_AGENT_ALIVE_STRING
        self.ws.send(json.dumps({'type': event_name, 'content': msg}))
        return msg['id']

    def send_message(self, text):
        data = {
            'text': text,
            'id': self.id,
            'message_id': str(uuid.uuid4()),
            'episode_done': False,
        }

        self.wants_to_send = False
        return self.build_and_send_packet(Packet.TYPE_MESSAGE, data)

    def send_alive(self):
        data = {
            'hit_id': self.hit_id,
            'assignment_id': self.assignment_id,
            'worker_id': self.worker_id,
            'conversation_id': self.conversation_id,
        }
        return self.build_and_send_packet(Packet.TYPE_ALIVE, data)

    def send_heartbeat(self):
        """
        Sends a heartbeat to the world.
        """
        hb = {
            'id': str(uuid.uuid4()),
            'receiver_id': '[World_' + self.task_group_id + ']',
            'assignment_id': self.assignment_id,
            'sender_id': self.worker_id,
            'conversation_id': self.conversation_id,
            'type': Packet.TYPE_HEARTBEAT,
            'data': None,
        }
        self.ws.send(
            json.dumps({'type': data_model.SOCKET_ROUTE_PACKET_STRING, 'content': hb})
        )

    def wait_for_alive(self):
        last_time = time.time()
        while not self.ready:
            self.send_alive()
            time.sleep(0.5)
            assert (
                time.time() - last_time < 10
            ), 'Timed out wating for server to acknowledge {} alive'.format(
                self.worker_id
            )


class TestMTurkManagerWorkflows(unittest.TestCase):
    """
    Various test cases to replicate a whole mturk workflow.
    """

    def setUp(self):
        patcher = mock.patch('builtins.input', return_value='y')
        self.addCleanup(patcher.stop)
        patcher.start()
        # Mock functions that hit external APIs and such
        self.server_utils = MTurkManagerFile.server_utils
        self.mturk_utils = MTurkManagerFile.mturk_utils
        self.server_utils.setup_server = mock.MagicMock(
            return_value='https://127.0.0.1'
        )
        self.server_utils.setup_legacy_server = mock.MagicMock(
            return_value='https://127.0.0.1'
        )
        self.server_utils.delete_server = mock.MagicMock()
        self.mturk_utils.setup_aws_credentials = mock.MagicMock()
        self.mturk_utils.calculate_mturk_cost = mock.MagicMock(return_value=1)
        self.mturk_utils.check_mturk_balance = mock.MagicMock(return_value=True)
        self.mturk_utils.create_hit_config = mock.MagicMock()
        self.mturk_utils.setup_sns_topic = mock.MagicMock(return_value=TOPIC_ARN)
        self.mturk_utils.delete_sns_topic = mock.MagicMock()
        self.mturk_utils.delete_qualification = mock.MagicMock()
        self.mturk_utils.find_or_create_qualification = mock.MagicMock(
            return_value=QUALIFICATION_ID
        )
        self.mturk_utils.find_qualification = mock.MagicMock(
            return_value=QUALIFICATION_ID
        )
        self.mturk_utils.give_worker_qualification = mock.MagicMock()
        self.mturk_utils.remove_worker_qualification = mock.MagicMock()
        self.mturk_utils.create_hit_type = mock.MagicMock(return_value=HIT_TYPE_ID)
        self.mturk_utils.subscribe_to_hits = mock.MagicMock()
        self.mturk_utils.create_hit_with_hit_type = mock.MagicMock(
            return_value=(MTURK_PAGE_URL, FAKE_HIT_ID, 'MTURK_HIT_DATA')
        )
        self.mturk_utils.get_mturk_client = mock.MagicMock(
            return_value=mock.MagicMock()
        )

        self.onboarding_agents = {}
        self.worlds_agents = {}

        # Set up an MTurk Manager and get it ready for accepting workers
        self.fake_socket = MockSocket()
        time.sleep(0.1)
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args(print_args=False)
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 1
        self.opt['hit_title'] = 'test_hit_title'
        self.opt['hit_description'] = 'test_hit_description'
        self.opt['task_description'] = 'test_task_description'
        self.opt['hit_keywords'] = 'test_hit_keywords'
        self.opt['reward'] = 0.1
        self.opt['is_debug'] = True
        self.opt['log_level'] = 0
        self.opt['num_conversations'] = 1
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(
            opt=self.opt, mturk_agent_ids=self.mturk_agent_ids, is_test=True
        )
        self.mturk_manager.port = self.fake_socket.port
        self.mturk_manager.setup_server()
        self.mturk_manager.start_new_run()
        self.mturk_manager.ready_to_accept_workers()
        self.mturk_manager.set_onboard_function(self.onboard_agent)
        self.mturk_manager.create_hits()

        def assign_worker_roles(workers):
            workers[0].id = 'mturk_agent_1'
            workers[1].id = 'mturk_agent_2'

        def run_task_wait():
            self.mturk_manager.start_task(
                lambda w: True, assign_worker_roles, self.run_conversation
            )

        self.task_thread = threading.Thread(target=run_task_wait)
        self.task_thread.start()

        self.agent_1 = MockAgent(
            TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1, TASK_GROUP_ID_1
        )
        self.agent_1_2 = MockAgent(
            TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_3, TEST_WORKER_ID_1, TASK_GROUP_ID_1
        )
        self.agent_2 = MockAgent(
            TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2, TASK_GROUP_ID_1
        )

    def tearDown(self):
        self.agent_1.always_beat = False
        self.agent_2.always_beat = False
        for key in self.worlds_agents.keys():
            self.worlds_agents[key] = True
        self.mturk_manager.shutdown()
        self.fake_socket.close()
        self.task_thread.join()

    def onboard_agent(self, worker):
        self.onboarding_agents[worker.worker_id] = False
        while (worker.worker_id in self.onboarding_agents) and (
            self.onboarding_agents[worker.worker_id] is False
        ):
            time.sleep(0.05)
        return

    def run_conversation(self, mturk_manager, opt, workers):
        for worker in workers:
            self.worlds_agents[worker.worker_id] = False
        for worker in workers:
            while self.worlds_agents[worker.worker_id] is False:
                time.sleep(0.05)
        for worker in workers:
            worker.shutdown(timeout=-1)

    def alive_agent(self, agent):
        agent.register_to_socket(self.fake_socket)
        agent.wait_for_alive()
        agent.send_heartbeat()

    def test_successful_convo(self):
        manager = self.mturk_manager

        # Alive first agent
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(
            agent_1.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_1.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)

        # Alive second agent
        agent_2 = self.agent_2
        self.alive_agent(agent_2)
        assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
        agent_2_object = manager.worker_manager.get_agent_for_assignment(
            agent_2.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_2.worker_id])
        self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_2.worker_id] = True
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_WAITING, 2)

        # Assert agents move to task
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)

        # Complete agents
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)

        # Assert conversation is complete for manager and agents
        assert_equal_by(lambda: manager.completed_conversations, 1, 2)
        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_1.message_packet
                    if p.data['text'] == data_model.COMMAND_SHOW_DONE_BUTTON
                ]
            ),
            1,
            2,
        )
        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_2.message_packet
                    if p.data['text'] == data_model.COMMAND_SHOW_DONE_BUTTON
                ]
            ),
            1,
            2,
        )

        # Assert sockets are closed
        assert_equal_by(
            lambda: len([x for x in manager.socket_manager.run.values() if not x]), 2, 2
        )

    def test_disconnect_end(self):
        manager = self.mturk_manager

        # Alive first agent
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(
            agent_1.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_1.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)

        # Alive second agent
        agent_2 = self.agent_2
        self.alive_agent(agent_2)
        assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
        agent_2_object = manager.worker_manager.get_agent_for_assignment(
            agent_2.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_2.worker_id])
        self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_2.worker_id] = True
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_WAITING, 2)

        # Assert agents move to task
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)

        # Disconnect agent
        agent_2.always_beat = False
        assert_equal_by(
            agent_1_object.get_status, AssignState.STATUS_PARTNER_DISCONNECT, 3
        )
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DISCONNECT, 3)
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        agent_2.always_beat = True
        agent_2.send_alive()

        # Assert workers get the correct command
        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_1.message_packet
                    if p.data['text'] == data_model.COMMAND_INACTIVE_DONE
                ]
            ),
            1,
            2,
        )
        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_2.message_packet
                    if p.data['text'] == data_model.COMMAND_INACTIVE_HIT
                ]
            ),
            1,
            2,
        )

        # assert conversation not marked as complete
        self.assertEqual(manager.completed_conversations, 0)

        # Assert sockets are closed
        assert_equal_by(
            lambda: len([x for x in manager.socket_manager.run.values() if not x]), 2, 2
        )

    def test_expire_onboarding(self):
        manager = self.mturk_manager

        # Alive first agent
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 10)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(
            agent_1.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)

        manager._expire_onboarding_pool()

        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_1.message_packet
                    if p.data['text'] == data_model.COMMAND_EXPIRE_HIT
                ]
            ),
            1,
            10,
        )

        self.onboarding_agents[agent_1.worker_id] = True

        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_EXPIRED)

        # Assert sockets are closed
        assert_equal_by(
            lambda: len([x for x in manager.socket_manager.run.values() if not x]),
            1,
            10,
        )

    def test_reconnect_complete(self):
        manager = self.mturk_manager

        # Alive first agent
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(
            agent_1.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_1.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)

        # Alive second agent
        agent_2 = self.agent_2
        self.alive_agent(agent_2)
        assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
        agent_2_object = manager.worker_manager.get_agent_for_assignment(
            agent_2.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_2.worker_id])
        self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_2.worker_id] = True
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_WAITING, 2)

        # Assert agents move to task
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)

        # Simulate reconnect to task
        stored_conv_id = agent_2.conversation_id
        stored_agent_id = agent_2.id
        agent_2.conversation_id = None
        agent_2.id = None
        agent_2.send_alive()

        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_2.message_packet
                    if p.data['text'] == data_model.COMMAND_RESTORE_STATE
                ]
            ),
            1,
            4,
        )
        self.assertEqual(agent_2.id, stored_agent_id)
        self.assertEqual(agent_2.conversation_id, stored_conv_id)

        # Complete agents
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)

        # Assert conversation is complete for manager and agents
        assert_equal_by(lambda: manager.completed_conversations, 1, 2)
        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_1.message_packet
                    if p.data['text'] == data_model.COMMAND_SHOW_DONE_BUTTON
                ]
            ),
            1,
            2,
        )
        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_2.message_packet
                    if p.data['text'] == data_model.COMMAND_SHOW_DONE_BUTTON
                ]
            ),
            1,
            2,
        )

        # Assert sockets are closed
        assert_equal_by(
            lambda: len([x for x in manager.socket_manager.run.values() if not x]), 2, 2
        )

    def test_attempt_break_unique(self):
        manager = self.mturk_manager
        unique_worker_qual = 'is_unique_qual'
        manager.is_unique = True
        manager.opt['unique_qual_name'] = unique_worker_qual
        manager.unique_qual_name = unique_worker_qual

        # Alive first agent
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(
            agent_1.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_1.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)

        # Alive second agent
        agent_2 = self.agent_2
        self.alive_agent(agent_2)
        assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
        agent_2_object = manager.worker_manager.get_agent_for_assignment(
            agent_2.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_2.worker_id])
        self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_2.worker_id] = True
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_WAITING, 2)

        # Assert agents move to task
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)

        # Complete agents
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)

        # Assert conversation is complete for manager and agents
        assert_equal_by(lambda: manager.completed_conversations, 1, 2)
        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_1.message_packet
                    if p.data['text'] == data_model.COMMAND_SHOW_DONE_BUTTON
                ]
            ),
            1,
            2,
        )
        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_2.message_packet
                    if p.data['text'] == data_model.COMMAND_SHOW_DONE_BUTTON
                ]
            ),
            1,
            2,
        )

        # Assert sockets are closed
        assert_equal_by(
            lambda: len([x for x in manager.socket_manager.run.values() if not x]), 2, 2
        )

        # ensure qualification was 'granted'
        self.mturk_utils.find_qualification.assert_called_with(
            unique_worker_qual, manager.is_sandbox
        )
        self.mturk_utils.give_worker_qualification.assert_any_call(
            agent_1.worker_id, QUALIFICATION_ID, None, manager.is_sandbox
        )
        self.mturk_utils.give_worker_qualification.assert_any_call(
            agent_2.worker_id, QUALIFICATION_ID, None, manager.is_sandbox
        )

        # Try to alive with the first agent a second time
        agent_1_2 = self.agent_1_2
        self.alive_agent(agent_1_2)
        assert_equal_by(lambda: agent_1_2.worker_id in self.onboarding_agents, True, 2)
        agent_1_2_object = manager.worker_manager.get_agent_for_assignment(
            agent_1_2.assignment_id
        )

        # No worker should be created for a unique task
        self.assertIsNone(agent_1_2_object)

        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_1_2.message_packet
                    if p.data['text'] == data_model.COMMAND_EXPIRE_HIT
                ]
            ),
            1,
            2,
        )

        # Assert sockets are closed
        assert_equal_by(
            lambda: len([x for x in manager.socket_manager.run.values() if not x]), 3, 2
        )

    def test_break_multi_convo(self):
        manager = self.mturk_manager
        manager.opt['allowed_conversations'] = 1

        # Alive first agent
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(
            agent_1.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_1.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)

        # Alive second agent
        agent_2 = self.agent_2
        self.alive_agent(agent_2)
        assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
        agent_2_object = manager.worker_manager.get_agent_for_assignment(
            agent_2.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_2.worker_id])
        self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_2.worker_id] = True
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_WAITING, 2)

        # Assert agents move to task
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)

        # Attempt to start a new conversation with duplicate worker 1
        agent_1_2 = self.agent_1_2
        self.alive_agent(agent_1_2)
        assert_equal_by(lambda: agent_1_2.worker_id in self.onboarding_agents, True, 2)
        agent_1_2_object = manager.worker_manager.get_agent_for_assignment(
            agent_1_2.assignment_id
        )

        # No worker should be created for a unique task
        self.assertIsNone(agent_1_2_object)

        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_1_2.message_packet
                    if p.data['text'] == data_model.COMMAND_EXPIRE_HIT
                ]
            ),
            1,
            2,
        )

        # Assert sockets are closed
        assert_equal_by(
            lambda: len([x for x in manager.socket_manager.run.values() if not x]), 1, 2
        )

        # Complete agents
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)

        # Assert conversation is complete for manager and agents
        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_1.message_packet
                    if p.data['text'] == data_model.COMMAND_SHOW_DONE_BUTTON
                ]
            ),
            1,
            2,
        )
        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_2.message_packet
                    if p.data['text'] == data_model.COMMAND_SHOW_DONE_BUTTON
                ]
            ),
            1,
            2,
        )
        assert_equal_by(lambda: manager.completed_conversations, 1, 2)

        # Assert sockets are closed
        assert_equal_by(
            lambda: len([x for x in manager.socket_manager.run.values() if not x]), 3, 2
        )

    def test_no_onboard_expire_waiting(self):
        manager = self.mturk_manager
        manager.set_onboard_function(None)

        # Alive first agent
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(
            agent_1.assignment_id
        )
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)

        manager._expire_agent_pool()

        assert_equal_by(
            lambda: len(
                [
                    p
                    for p in agent_1.message_packet
                    if p.data['text'] == data_model.COMMAND_EXPIRE_HIT
                ]
            ),
            1,
            2,
        )

        # Assert sockets are closed
        assert_equal_by(
            lambda: len([x for x in manager.socket_manager.run.values() if not x]), 1, 2
        )

    def test_return_to_waiting_on_world_start(self):
        manager = self.mturk_manager

        # Alive first agent
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(
            agent_1.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_1.worker_id])
        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_1.worker_id] = True
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)

        # Make agent_1 no longer respond to change_conversation_requests
        def replace_on_msg(packet):
            agent_1.message_packet.append(packet)

        agent_1.on_msg = replace_on_msg

        # Alive second agent
        agent_2 = self.agent_2
        self.alive_agent(agent_2)
        assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
        agent_2_object = manager.worker_manager.get_agent_for_assignment(
            agent_2.assignment_id
        )
        self.assertFalse(self.onboarding_agents[agent_2.worker_id])
        self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
        self.onboarding_agents[agent_2.worker_id] = True
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_WAITING, 2)

        # Assert agents attempt to move to task, but then move back to waiting
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_WAITING, 3)
        agent_1.always_beat = False

        # Assert no world ever started
        self.assertNotIn(agent_2.worker_id, self.worlds_agents)

        # Expire everything
        manager.shutdown()

        # Assert sockets are closed
        assert_equal_by(
            lambda: len([x for x in manager.socket_manager.run.values() if not x]), 2, 2
        )


if __name__ == '__main__':
    unittest.main(buffer=True)
