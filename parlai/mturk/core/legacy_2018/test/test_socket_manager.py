#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.agents import AssignState

import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json

TEST_WORKER_ID_1 = 'TEST_WORKER_ID_1'
TEST_ASSIGNMENT_ID_1 = 'TEST_ASSIGNMENT_ID_1'
TEST_HIT_ID_1 = 'TEST_HIT_ID_1'
TEST_WORKER_ID_2 = 'TEST_WORKER_ID_2'
TEST_ASSIGNMENT_ID_2 = 'TEST_ASSIGNMENT_ID_2'
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
    AssignState.STATUS_NONE, AssignState.STATUS_ONBOARDING,
    AssignState.STATUS_WAITING, AssignState.STATUS_IN_TASK,
]
complete_statuses = [
    AssignState.STATUS_DONE, AssignState.STATUS_DISCONNECT,
    AssignState.STATUS_PARTNER_DISCONNECT,
    AssignState.STATUS_PARTNER_DISCONNECT_EARLY,
    AssignState.STATUS_EXPIRED, AssignState.STATUS_RETURNED,
]
statuses = active_statuses + complete_statuses

TASK_GROUP_ID_1 = 'TASK_GROUP_ID_1'

SocketManager.DEF_MISSED_PONGS = 3
SocketManager.HEARTBEAT_RATE = 0.6
SocketManager.DEF_DEAD_TIME = 0.6
SocketManager.ACK_TIME = {Packet.TYPE_ALIVE: 0.4,
                          Packet.TYPE_MESSAGE: 0.2}

shared_utils.THREAD_SHORT_SLEEP = 0.05
shared_utils.THREAD_MEDIUM_SLEEP = 0.15


class TestPacket(unittest.TestCase):
    """Various unit tests for the AssignState class"""

    ID = 'ID'
    SENDER_ID = 'SENDER_ID'
    RECEIVER_ID = 'RECEIVER_ID'
    ASSIGNMENT_ID = 'ASSIGNMENT_ID'
    DATA = 'DATA'
    CONVERSATION_ID = 'CONVERSATION_ID'
    REQUIRES_ACK = True
    BLOCKING = False
    ACK_FUNCTION = 'ACK_FUNCTION'

    def setUp(self):
        self.packet_1 = Packet(self.ID, Packet.TYPE_MESSAGE, self.SENDER_ID,
                               self.RECEIVER_ID, self.ASSIGNMENT_ID, self.DATA,
                               conversation_id=self.CONVERSATION_ID,
                               requires_ack=self.REQUIRES_ACK,
                               blocking=self.BLOCKING,
                               ack_func=self.ACK_FUNCTION)
        self.packet_2 = Packet(self.ID, Packet.TYPE_HEARTBEAT, self.SENDER_ID,
                               self.RECEIVER_ID, self.ASSIGNMENT_ID, self.DATA)
        self.packet_3 = Packet(self.ID, Packet.TYPE_ALIVE, self.SENDER_ID,
                               self.RECEIVER_ID, self.ASSIGNMENT_ID, self.DATA)

    def tearDown(self):
        pass

    def test_packet_init(self):
        '''Test proper initialization of packet fields'''
        self.assertEqual(self.packet_1.id, self.ID)
        self.assertEqual(self.packet_1.type, Packet.TYPE_MESSAGE)
        self.assertEqual(self.packet_1.sender_id, self.SENDER_ID)
        self.assertEqual(self.packet_1.receiver_id, self.RECEIVER_ID)
        self.assertEqual(self.packet_1.assignment_id, self.ASSIGNMENT_ID)
        self.assertEqual(self.packet_1.data, self.DATA)
        self.assertEqual(self.packet_1.conversation_id, self.CONVERSATION_ID)
        self.assertEqual(self.packet_1.requires_ack, self.REQUIRES_ACK)
        self.assertEqual(self.packet_1.blocking, self.BLOCKING)
        self.assertEqual(self.packet_1.ack_func, self.ACK_FUNCTION)
        self.assertEqual(self.packet_1.status, Packet.STATUS_INIT)
        self.assertEqual(self.packet_2.id, self.ID)
        self.assertEqual(self.packet_2.type, Packet.TYPE_HEARTBEAT)
        self.assertEqual(self.packet_2.sender_id, self.SENDER_ID)
        self.assertEqual(self.packet_2.receiver_id, self.RECEIVER_ID)
        self.assertEqual(self.packet_2.assignment_id, self.ASSIGNMENT_ID)
        self.assertEqual(self.packet_2.data, self.DATA)
        self.assertIsNone(self.packet_2.conversation_id)
        self.assertFalse(self.packet_2.requires_ack)
        self.assertFalse(self.packet_2.blocking)
        self.assertIsNone(self.packet_2.ack_func)
        self.assertEqual(self.packet_2.status, Packet.STATUS_INIT)
        self.assertEqual(self.packet_3.id, self.ID)
        self.assertEqual(self.packet_3.type, Packet.TYPE_ALIVE)
        self.assertEqual(self.packet_3.sender_id, self.SENDER_ID)
        self.assertEqual(self.packet_3.receiver_id, self.RECEIVER_ID)
        self.assertEqual(self.packet_3.assignment_id, self.ASSIGNMENT_ID)
        self.assertEqual(self.packet_3.data, self.DATA)
        self.assertIsNone(self.packet_3.conversation_id)
        self.assertTrue(self.packet_3.requires_ack)
        self.assertTrue(self.packet_3.blocking)
        self.assertIsNone(self.packet_3.ack_func)
        self.assertEqual(self.packet_3.status, Packet.STATUS_INIT)

    def test_dict_conversion(self):
        '''Ensure packets can be converted to and from a representative dict'''
        converted_packet = Packet.from_dict(self.packet_1.as_dict())
        self.assertEqual(self.packet_1.id, converted_packet.id)
        self.assertEqual(self.packet_1.type, converted_packet.type)
        self.assertEqual(
            self.packet_1.sender_id, converted_packet.sender_id)
        self.assertEqual(
            self.packet_1.receiver_id, converted_packet.receiver_id)
        self.assertEqual(
            self.packet_1.assignment_id, converted_packet.assignment_id)
        self.assertEqual(self.packet_1.data, converted_packet.data)
        self.assertEqual(
            self.packet_1.conversation_id, converted_packet.conversation_id)

        packet_dict = self.packet_1.as_dict()
        self.assertDictEqual(
            packet_dict, Packet.from_dict(packet_dict).as_dict())

    def test_connection_ids(self):
        '''Ensure that connection ids are reported as we expect them'''
        sender_conn_id = '{}_{}'.format(self.SENDER_ID, self.ASSIGNMENT_ID)
        receiver_conn_id = '{}_{}'.format(self.RECEIVER_ID, self.ASSIGNMENT_ID)
        self.assertEqual(
            self.packet_1.get_sender_connection_id(), sender_conn_id)
        self.assertEqual(
            self.packet_1.get_receiver_connection_id(), receiver_conn_id)

    def test_packet_conversions(self):
        '''Ensure that packet copies and acts are produced properly'''
        # Copy important packet
        message_packet_copy = self.packet_1.new_copy()
        self.assertNotEqual(message_packet_copy.id, self.ID)
        self.assertNotEqual(message_packet_copy, self.packet_1)
        self.assertEqual(message_packet_copy.type, self.packet_1.type)
        self.assertEqual(
            message_packet_copy.sender_id, self.packet_1.sender_id)
        self.assertEqual(
            message_packet_copy.receiver_id, self.packet_1.receiver_id)
        self.assertEqual(
            message_packet_copy.assignment_id, self.packet_1.assignment_id)
        self.assertEqual(message_packet_copy.data, self.packet_1.data)
        self.assertEqual(
            message_packet_copy.conversation_id, self.packet_1.conversation_id)
        self.assertEqual(
            message_packet_copy.requires_ack, self.packet_1.requires_ack)
        self.assertEqual(
            message_packet_copy.blocking, self.packet_1.blocking)
        self.assertIsNone(message_packet_copy.ack_func)
        self.assertEqual(message_packet_copy.status, Packet.STATUS_INIT)

        # Copy non-important packet
        hb_packet_copy = self.packet_2.new_copy()
        self.assertNotEqual(hb_packet_copy.id, self.ID)
        self.assertNotEqual(hb_packet_copy, self.packet_2)
        self.assertEqual(hb_packet_copy.type, self.packet_2.type)
        self.assertEqual(hb_packet_copy.sender_id, self.packet_2.sender_id)
        self.assertEqual(hb_packet_copy.receiver_id, self.packet_2.receiver_id)
        self.assertEqual(
            hb_packet_copy.assignment_id, self.packet_2.assignment_id)
        self.assertEqual(hb_packet_copy.data, self.packet_2.data)
        self.assertEqual(
            hb_packet_copy.conversation_id, self.packet_2.conversation_id)
        self.assertEqual(
            hb_packet_copy.requires_ack, self.packet_2.requires_ack)
        self.assertEqual(hb_packet_copy.blocking, self.packet_2.blocking)
        self.assertIsNone(hb_packet_copy.ack_func)
        self.assertEqual(hb_packet_copy.status, Packet.STATUS_INIT)

        # ack important packet
        ack_packet = self.packet_1.get_ack()
        self.assertEqual(ack_packet.id, self.ID)
        self.assertEqual(ack_packet.type, Packet.TYPE_ACK)
        self.assertEqual(ack_packet.sender_id, self.RECEIVER_ID)
        self.assertEqual(ack_packet.receiver_id, self.SENDER_ID)
        self.assertEqual(ack_packet.assignment_id, self.ASSIGNMENT_ID)
        self.assertEqual(ack_packet.data, '')
        self.assertEqual(ack_packet.conversation_id, self.CONVERSATION_ID)
        self.assertFalse(ack_packet.requires_ack)
        self.assertFalse(ack_packet.blocking)
        self.assertIsNone(ack_packet.ack_func)
        self.assertEqual(ack_packet.status, Packet.STATUS_INIT)

    def test_packet_modifications(self):
        '''Ensure that packet copies and acts are produced properly'''
        # All operations return the packet
        self.assertEqual(self.packet_1.swap_sender(), self.packet_1)
        self.assertEqual(
            self.packet_1.set_type(Packet.TYPE_ACK), self.packet_1)
        self.assertEqual(self.packet_1.set_data(None), self.packet_1)

        # Ensure all of the operations worked
        self.assertEqual(self.packet_1.sender_id, self.RECEIVER_ID)
        self.assertEqual(self.packet_1.receiver_id, self.SENDER_ID)
        self.assertEqual(self.packet_1.type, Packet.TYPE_ACK)
        self.assertIsNone(self.packet_1.data)


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


class MockAgent(object):
    """Class that pretends to be an MTurk agent interacting through the
    webpage by simulating the same commands that are sent from the core.html
    file. Exposes methods to use for testing and checking status
    """
    def __init__(self, hit_id, assignment_id, worker_id,
                 task_group_id):
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

    def send_packet(self, packet):
        def callback(*args):
            pass
        event_name = data_model.SOCKET_ROUTE_PACKET_STRING
        self.ws.send(json.dumps({
            'type': event_name,
            'content': packet.as_dict(),
        }))

    def register_to_socket(self, ws, on_ack, on_hb, on_msg):
        handler = self.make_packet_handler(on_ack, on_hb, on_msg)
        self.ws = ws
        self.ws.handlers[self.worker_id] = handler

    def make_packet_handler(self, on_ack, on_hb, on_msg):
        """A packet handler that properly sends heartbeats"""
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
                on_msg(packet)
            elif pkt['type'] == Packet.TYPE_ALIVE:
                raise Exception('Invalid alive packet {}'.format(pkt))
            else:
                raise Exception('Invalid Packet type {} received in {}'.format(
                    pkt['type'],
                    pkt
                ))
        return handler_mock

    def build_and_send_packet(self, packet_type, data):
        msg = {
            'id': str(uuid.uuid4()),
            'type': packet_type,
            'sender_id': self.worker_id,
            'assignment_id': self.assignment_id,
            'conversation_id': self.conversation_id,
            'receiver_id': '[World_' + self.task_group_id + ']',
            'data': data
        }

        event_name = data_model.SOCKET_ROUTE_PACKET_STRING
        if (packet_type == Packet.TYPE_ALIVE):
            event_name = data_model.SOCKET_AGENT_ALIVE_STRING
        self.ws.send(json.dumps({
            'type': event_name,
            'content': msg,
        }))
        return msg['id']

    def send_message(self, text):
        data = {
            'text': text,
            'id': self.id,
            'message_id': str(uuid.uuid4()),
            'episode_done': False
        }

        self.wants_to_send = False
        return self.build_and_send_packet(Packet.TYPE_MESSAGE, data)

    def send_alive(self):
        data = {
            'hit_id': self.hit_id,
            'assignment_id': self.assignment_id,
            'worker_id': self.worker_id,
            'conversation_id': self.conversation_id
        }
        return self.build_and_send_packet(Packet.TYPE_ALIVE, data)

    def send_heartbeat(self):
        """Sends a heartbeat to the world"""
        hb = {
            'id': str(uuid.uuid4()),
            'receiver_id': '[World_' + self.task_group_id + ']',
            'assignment_id': self.assignment_id,
            'sender_id': self.worker_id,
            'conversation_id': self.conversation_id,
            'type': Packet.TYPE_HEARTBEAT,
            'data': None
        }
        self.ws.send(json.dumps({
            'type': data_model.SOCKET_ROUTE_PACKET_STRING,
            'content': hb,
        }))

    def wait_for_alive(self):
        last_time = time.time()
        while not self.ready:
            self.send_alive()
            time.sleep(0.5)
            assert time.time() - last_time < 10, \
                'Timed out wating for server to acknowledge {} alive'.format(
                    self.worker_id
            )


class TestSocketManagerSetupAndFunctions(unittest.TestCase):
    """Unit/integration tests for starting up a socket"""

    def setUp(self):
        self.fake_socket = MockSocket()
        time.sleep(1)

    def tearDown(self):
        self.fake_socket.close()

    def test_init_and_reg_shutdown(self):
        '''Test initialization of a socket manager'''
        self.assertFalse(self.fake_socket.connected)

        # Callbacks should never trigger during proper setup and shutdown
        nop_called = False

        def nop(*args):
            nonlocal nop_called  # noqa 999 we don't support py2
            nop_called = True

        socket_manager = SocketManager('https://127.0.0.1',
                                       self.fake_socket.port, nop, nop,
                                       nop, TASK_GROUP_ID_1, 0.3, nop)
        self.assertTrue(self.fake_socket.connected)
        self.assertFalse(nop_called)

        # Test shutdown
        self.assertFalse(self.fake_socket.disconnected)
        self.assertFalse(socket_manager.is_shutdown)
        self.assertTrue(socket_manager.alive)
        socket_manager.shutdown()
        self.assertTrue(self.fake_socket.disconnected)
        self.assertTrue(socket_manager.is_shutdown)
        self.assertFalse(nop_called)

    def assertEqualBy(self, val_func, val, max_time):
        start_time = time.time()
        while val_func() != val:
            assert time.time() - start_time < max_time, \
                "Value was not attained in specified time, was {} rather " \
                "than {}".format(val_func(), val)
            time.sleep(0.1)

    def test_init_and_socket_shutdown(self):
        '''Test initialization of a socket manager with a failed shutdown'''
        self.assertFalse(self.fake_socket.connected)

        # Callbacks should never trigger during proper setup and shutdown
        nop_called = False

        def nop(*args):
            nonlocal nop_called  # noqa 999 we don't support py2
            nop_called = True

        server_death_called = False

        def server_death(*args):
            nonlocal server_death_called
            server_death_called = True

        socket_manager = SocketManager('https://127.0.0.1',
                                       self.fake_socket.port, nop, nop,
                                       nop, TASK_GROUP_ID_1, 0.4, server_death)
        self.assertTrue(self.fake_socket.connected)
        self.assertFalse(nop_called)
        self.assertFalse(server_death_called)

        # Test shutdown
        self.assertFalse(self.fake_socket.disconnected)
        self.assertFalse(socket_manager.is_shutdown)
        self.assertTrue(socket_manager.alive)
        self.fake_socket.close()
        self.assertEqualBy(lambda: socket_manager.alive, False,
                           8 * socket_manager.HEARTBEAT_RATE)
        self.assertEqualBy(lambda: server_death_called, True,
                           4 * socket_manager.HEARTBEAT_RATE)
        self.assertFalse(nop_called)
        socket_manager.shutdown()

    def test_init_and_socket_shutdown_then_restart(self):
        '''Test restoring connection to a socket'''
        self.assertFalse(self.fake_socket.connected)

        # Callbacks should never trigger during proper setup and shutdown
        nop_called = False

        def nop(*args):
            nonlocal nop_called  # noqa 999 we don't support py2
            nop_called = True

        server_death_called = False

        def server_death(*args):
            nonlocal server_death_called
            server_death_called = True

        socket_manager = SocketManager('https://127.0.0.1',
                                       self.fake_socket.port, nop, nop,
                                       nop, TASK_GROUP_ID_1, 0.4, server_death)
        self.assertTrue(self.fake_socket.connected)
        self.assertFalse(nop_called)
        self.assertFalse(server_death_called)

        # Test shutdown
        self.assertFalse(self.fake_socket.disconnected)
        self.assertFalse(socket_manager.is_shutdown)
        self.assertTrue(socket_manager.alive)
        self.fake_socket.close()
        self.assertEqualBy(lambda: socket_manager.alive, False,
                           8 * socket_manager.HEARTBEAT_RATE)
        self.assertFalse(socket_manager.alive)
        self.fake_socket = MockSocket()
        self.assertEqualBy(lambda: socket_manager.alive, True,
                           4 * socket_manager.HEARTBEAT_RATE)
        self.assertFalse(nop_called)
        self.assertFalse(server_death_called)
        socket_manager.shutdown()

    def test_init_world_dead(self):
        '''Test initialization of a socket manager with a failed startup'''
        self.assertFalse(self.fake_socket.connected)
        self.fake_socket.close()

        # Callbacks should never trigger during proper setup and shutdown
        nop_called = False

        def nop(*args):
            nonlocal nop_called  # noqa 999 we don't support py2
            nop_called = True

        server_death_called = False

        def server_death(*args):
            nonlocal server_death_called
            server_death_called = True

        with self.assertRaises(ConnectionRefusedError):
            socket_manager = SocketManager('https://127.0.0.1',
                                           self.fake_socket.port, nop, nop,
                                           nop, TASK_GROUP_ID_1, 0.4,
                                           server_death)
            self.assertIsNone(socket_manager)

        self.assertFalse(nop_called)
        self.assertTrue(server_death_called)


class TestSocketManagerRoutingFunctionality(unittest.TestCase):

    ID = 'ID'
    SENDER_ID = 'SENDER_ID'
    ASSIGNMENT_ID = 'ASSIGNMENT_ID'
    DATA = 'DATA'
    CONVERSATION_ID = 'CONVERSATION_ID'
    REQUIRES_ACK = True
    BLOCKING = False
    ACK_FUNCTION = 'ACK_FUNCTION'
    WORLD_ID = '[World_{}]'.format(TASK_GROUP_ID_1)

    def on_alive(self, packet):
        self.alive_packet = packet

    def on_message(self, packet):
        self.message_packet = packet

    def on_worker_death(self, worker_id, assignment_id):
        self.dead_worker_id = worker_id
        self.dead_assignment_id = assignment_id

    def on_server_death(self):
        self.server_died = True

    def setUp(self):
        self.AGENT_HEARTBEAT_PACKET = Packet(
            self.ID, Packet.TYPE_HEARTBEAT, self.SENDER_ID, self.WORLD_ID,
            self.ASSIGNMENT_ID, self.DATA, self.CONVERSATION_ID)

        self.AGENT_ALIVE_PACKET = Packet(
            MESSAGE_ID_1, Packet.TYPE_ALIVE, self.SENDER_ID, self.WORLD_ID,
            self.ASSIGNMENT_ID, self.DATA, self.CONVERSATION_ID)

        self.MESSAGE_SEND_PACKET_1 = Packet(
            MESSAGE_ID_2, Packet.TYPE_MESSAGE, self.WORLD_ID, self.SENDER_ID,
            self.ASSIGNMENT_ID, self.DATA, self.CONVERSATION_ID)

        self.MESSAGE_SEND_PACKET_2 = Packet(
            MESSAGE_ID_3, Packet.TYPE_MESSAGE, self.WORLD_ID, self.SENDER_ID,
            self.ASSIGNMENT_ID, self.DATA, self.CONVERSATION_ID,
            requires_ack=False)

        self.MESSAGE_SEND_PACKET_3 = Packet(
            MESSAGE_ID_4, Packet.TYPE_MESSAGE, self.WORLD_ID, self.SENDER_ID,
            self.ASSIGNMENT_ID, self.DATA, self.CONVERSATION_ID,
            blocking=False)

        self.fake_socket = MockSocket()
        time.sleep(0.3)
        self.alive_packet = None
        self.message_packet = None
        self.dead_worker_id = None
        self.dead_assignment_id = None
        self.server_died = False

        self.socket_manager = SocketManager(
            'https://127.0.0.1', self.fake_socket.port, self.on_alive,
            self.on_message, self.on_worker_death, TASK_GROUP_ID_1, 1,
            self.on_server_death)

    def tearDown(self):
        self.socket_manager.shutdown()
        self.fake_socket.close()

    def test_init_state(self):
        '''Ensure all of the initial state of the socket_manager is ready'''
        self.assertEqual(self.socket_manager.server_url, 'https://127.0.0.1')
        self.assertEqual(self.socket_manager.port, self.fake_socket.port)
        self.assertEqual(self.socket_manager.alive_callback, self.on_alive)
        self.assertEqual(self.socket_manager.message_callback, self.on_message)
        self.assertEqual(self.socket_manager.socket_dead_callback,
                         self.on_worker_death)
        self.assertEqual(self.socket_manager.task_group_id, TASK_GROUP_ID_1)
        self.assertEqual(self.socket_manager.missed_pongs,
                         1 + (1 / SocketManager.HEARTBEAT_RATE))
        self.assertIsNotNone(self.socket_manager.ws)
        self.assertTrue(self.socket_manager.keep_running)
        self.assertIsNotNone(self.socket_manager.listen_thread)
        self.assertDictEqual(self.socket_manager.queues, {})
        self.assertDictEqual(self.socket_manager.threads, {})
        self.assertDictEqual(self.socket_manager.run, {})
        self.assertDictEqual(self.socket_manager.last_sent_heartbeat_time, {})
        self.assertDictEqual(self.socket_manager.last_received_heartbeat, {})
        self.assertDictEqual(self.socket_manager.pongs_without_heartbeat, {})
        self.assertDictEqual(self.socket_manager.packet_map, {})
        self.assertTrue(self.socket_manager.alive)
        self.assertFalse(self.socket_manager.is_shutdown)
        self.assertEqual(self.socket_manager.get_my_sender_id(), self.WORLD_ID)

    def test_needed_heartbeat(self):
        '''Ensure needed heartbeat sends heartbeats at the right time'''
        self.socket_manager._safe_send = mock.MagicMock()
        connection_id = self.AGENT_HEARTBEAT_PACKET.get_sender_connection_id()

        # Ensure no failure under uninitialized cases
        self.socket_manager._send_needed_heartbeat(connection_id)
        self.socket_manager.last_received_heartbeat[connection_id] = None
        self.socket_manager._send_needed_heartbeat(connection_id)

        self.socket_manager._safe_send.assert_not_called()

        # assert not called when called too recently
        self.socket_manager.last_received_heartbeat[connection_id] = \
            self.AGENT_HEARTBEAT_PACKET
        self.socket_manager.last_sent_heartbeat_time[connection_id] = \
            time.time() + 10

        self.socket_manager._send_needed_heartbeat(connection_id)

        self.socket_manager._safe_send.assert_not_called()

        # Assert called when supposed to
        self.socket_manager.last_sent_heartbeat_time[connection_id] = \
            time.time() - SocketManager.HEARTBEAT_RATE
        self.assertGreater(
            time.time() -
            self.socket_manager.last_sent_heartbeat_time[connection_id],
            SocketManager.HEARTBEAT_RATE)
        self.socket_manager._send_needed_heartbeat(connection_id)
        self.assertLess(
            time.time() -
            self.socket_manager.last_sent_heartbeat_time[connection_id],
            SocketManager.HEARTBEAT_RATE)
        used_packet_json = self.socket_manager._safe_send.call_args[0][0]
        used_packet_dict = json.loads(used_packet_json)
        self.assertEqual(
            used_packet_dict['type'], data_model.SOCKET_ROUTE_PACKET_STRING)
        used_packet = Packet.from_dict(used_packet_dict['content'])
        self.assertNotEqual(self.AGENT_HEARTBEAT_PACKET.id, used_packet.id)
        self.assertEqual(used_packet.type, Packet.TYPE_HEARTBEAT)
        self.assertEqual(used_packet.sender_id, self.WORLD_ID)
        self.assertEqual(used_packet.receiver_id, self.SENDER_ID)
        self.assertEqual(used_packet.assignment_id, self.ASSIGNMENT_ID)
        self.assertEqual(used_packet.data, '')
        self.assertEqual(used_packet.conversation_id, self.CONVERSATION_ID)
        self.assertEqual(used_packet.requires_ack, False)
        self.assertEqual(used_packet.blocking, False)

    def test_ack_send(self):
        '''Ensure acks are being properly created and sent'''
        self.socket_manager._safe_send = mock.MagicMock()
        self.socket_manager._send_ack(self.AGENT_ALIVE_PACKET)
        used_packet_json = self.socket_manager._safe_send.call_args[0][0]
        used_packet_dict = json.loads(used_packet_json)
        self.assertEqual(
            used_packet_dict['type'], data_model.SOCKET_ROUTE_PACKET_STRING)
        used_packet = Packet.from_dict(used_packet_dict['content'])
        self.assertEqual(self.AGENT_ALIVE_PACKET.id, used_packet.id)
        self.assertEqual(used_packet.type, Packet.TYPE_ACK)
        self.assertEqual(used_packet.sender_id, self.WORLD_ID)
        self.assertEqual(used_packet.receiver_id, self.SENDER_ID)
        self.assertEqual(used_packet.assignment_id, self.ASSIGNMENT_ID)
        self.assertEqual(used_packet.conversation_id, self.CONVERSATION_ID)
        self.assertEqual(used_packet.requires_ack, False)
        self.assertEqual(used_packet.blocking, False)
        self.assertEqual(self.AGENT_ALIVE_PACKET.status, Packet.STATUS_SENT)

    def _send_packet_in_background(self, packet, send_time):
        '''creates a thread to handle waiting for a packet send'''
        def do_send():
            self.socket_manager._send_packet(
                packet, packet.get_receiver_connection_id(), send_time
            )
            self.sent = True

        send_thread = threading.Thread(target=do_send, daemon=True)
        send_thread.start()
        time.sleep(0.02)

    def test_blocking_ack_packet_send(self):
        '''Checks to see if ack'ed blocking packets are working properly'''
        self.socket_manager._safe_send = mock.MagicMock()
        self.socket_manager._safe_put = mock.MagicMock()
        self.sent = False

        # Test a blocking acknowledged packet
        send_time = time.time()
        self.assertEqual(self.MESSAGE_SEND_PACKET_1.status, Packet.STATUS_INIT)
        self._send_packet_in_background(self.MESSAGE_SEND_PACKET_1, send_time)
        self.assertEqual(self.MESSAGE_SEND_PACKET_1.status, Packet.STATUS_SENT)
        self.socket_manager._safe_send.assert_called_once()

        connection_id = self.MESSAGE_SEND_PACKET_1.get_receiver_connection_id()
        self.socket_manager._safe_put.assert_called_once_with(
            connection_id, (send_time, self.MESSAGE_SEND_PACKET_1))
        self.assertTrue(self.sent)

        self.socket_manager._safe_send.reset_mock()
        self.socket_manager._safe_put.reset_mock()

        # Send it again - end outcome should be a call to send only
        # with sent set
        self.MESSAGE_SEND_PACKET_1.status = Packet.STATUS_ACK
        self._send_packet_in_background(self.MESSAGE_SEND_PACKET_1, send_time)
        self.socket_manager._safe_send.assert_not_called()
        self.socket_manager._safe_put.assert_not_called()

    def test_non_blocking_ack_packet_send(self):
        '''Checks to see if ack'ed non-blocking packets are working'''
        self.socket_manager._safe_send = mock.MagicMock()
        self.socket_manager._safe_put = mock.MagicMock()
        self.sent = False

        # Test a blocking acknowledged packet
        send_time = time.time()
        self.assertEqual(self.MESSAGE_SEND_PACKET_3.status, Packet.STATUS_INIT)
        self._send_packet_in_background(self.MESSAGE_SEND_PACKET_3, send_time)
        self.assertEqual(self.MESSAGE_SEND_PACKET_3.status, Packet.STATUS_SENT)
        self.socket_manager._safe_send.assert_called_once()
        self.socket_manager._safe_put.assert_called_once()
        self.assertTrue(self.sent)

        call_args = self.socket_manager._safe_put.call_args[0]
        connection_id = call_args[0]
        queue_item = call_args[1]
        self.assertEqual(
            connection_id,
            self.MESSAGE_SEND_PACKET_3.get_receiver_connection_id())
        expected_send_time = \
            send_time + SocketManager.ACK_TIME[self.MESSAGE_SEND_PACKET_3.type]
        self.assertAlmostEqual(queue_item[0], expected_send_time, places=2)
        self.assertEqual(queue_item[1], self.MESSAGE_SEND_PACKET_3)
        used_packet_json = self.socket_manager._safe_send.call_args[0][0]
        used_packet_dict = json.loads(used_packet_json)
        self.assertEqual(
            used_packet_dict['type'], data_model.SOCKET_ROUTE_PACKET_STRING)
        self.assertDictEqual(used_packet_dict['content'],
                             self.MESSAGE_SEND_PACKET_3.as_dict())

    def test_non_ack_packet_send(self):
        '''Checks to see if non-ack'ed packets are working'''
        self.socket_manager._safe_send = mock.MagicMock()
        self.socket_manager._safe_put = mock.MagicMock()
        self.sent = False

        # Test a blocking acknowledged packet
        send_time = time.time()
        self.assertEqual(self.MESSAGE_SEND_PACKET_2.status, Packet.STATUS_INIT)
        self._send_packet_in_background(self.MESSAGE_SEND_PACKET_2, send_time)
        self.assertEqual(self.MESSAGE_SEND_PACKET_2.status, Packet.STATUS_SENT)
        self.socket_manager._safe_send.assert_called_once()
        self.socket_manager._safe_put.assert_not_called()
        self.assertTrue(self.sent)

        used_packet_json = self.socket_manager._safe_send.call_args[0][0]
        used_packet_dict = json.loads(used_packet_json)
        self.assertEqual(
            used_packet_dict['type'], data_model.SOCKET_ROUTE_PACKET_STRING)
        self.assertDictEqual(used_packet_dict['content'],
                             self.MESSAGE_SEND_PACKET_2.as_dict())

    def test_simple_packet_channel_management(self):
        '''Ensure that channels are created, managed, and then removed
        as expected
        '''
        self.socket_manager._safe_put = mock.MagicMock()
        use_packet = self.MESSAGE_SEND_PACKET_1
        worker_id = use_packet.receiver_id
        assignment_id = use_packet.assignment_id

        # Open a channel and assert it is there
        self.socket_manager.open_channel(worker_id, assignment_id)
        time.sleep(0.1)
        connection_id = use_packet.get_receiver_connection_id()
        self.assertTrue(self.socket_manager.run[connection_id])

        self.assertIsNotNone(self.socket_manager.queues[connection_id])
        self.assertEqual(
            self.socket_manager.last_sent_heartbeat_time[connection_id], 0)
        self.assertEqual(
            self.socket_manager.pongs_without_heartbeat[connection_id], 0)
        self.assertIsNone(
            self.socket_manager.last_received_heartbeat[connection_id])
        self.assertTrue(self.socket_manager.socket_is_open(connection_id))
        self.assertFalse(self.socket_manager.socket_is_open(FAKE_ID))

        # Send a bad packet, ensure it is ignored
        resp = self.socket_manager.queue_packet(self.AGENT_ALIVE_PACKET)
        self.socket_manager._safe_put.assert_not_called()
        self.assertFalse(resp)
        self.assertNotIn(self.AGENT_ALIVE_PACKET.id,
                         self.socket_manager.packet_map)

        # Send a packet to an open socket, ensure it got queued
        resp = self.socket_manager.queue_packet(use_packet)
        self.socket_manager._safe_put.assert_called_once()
        self.assertIn(use_packet.id, self.socket_manager.packet_map)
        self.assertTrue(resp)

        # Assert we can get the status of a packet in the map, but not
        # existing doesn't throw an error
        self.assertEqual(self.socket_manager.get_status(use_packet.id),
                         use_packet.status)
        self.assertEqual(self.socket_manager.get_status(FAKE_ID),
                         Packet.STATUS_NONE)

        # Assert that closing a thread does the correct cleanup work
        self.socket_manager.close_channel(connection_id)
        time.sleep(0.2)
        self.assertFalse(self.socket_manager.run[connection_id])
        self.assertNotIn(connection_id, self.socket_manager.queues)
        self.assertNotIn(connection_id, self.socket_manager.threads)
        self.assertNotIn(use_packet.id, self.socket_manager.packet_map)

        # Assert that opening multiple threads and closing them is possible
        self.socket_manager.open_channel(worker_id, assignment_id)
        self.socket_manager.open_channel(worker_id + '2', assignment_id)
        time.sleep(0.1)
        self.assertEqual(len(self.socket_manager.queues), 2)
        self.socket_manager.close_all_channels()
        time.sleep(0.1)
        self.assertEqual(len(self.socket_manager.queues), 0)

    def test_safe_put(self):
        '''Test safe put and queue retrieval mechanisms'''
        self.socket_manager._send_packet = mock.MagicMock()
        use_packet = self.MESSAGE_SEND_PACKET_1
        worker_id = use_packet.receiver_id
        assignment_id = use_packet.assignment_id
        connection_id = use_packet.get_receiver_connection_id()

        # Open a channel and assert it is there
        self.socket_manager.open_channel(worker_id, assignment_id)
        send_time = time.time()
        self.socket_manager._safe_put(connection_id, (send_time, use_packet))

        # Wait for the sending thread to try to pull the packet from the queue
        time.sleep(0.3)

        # Ensure the right packet was popped and sent.
        self.socket_manager._send_packet.assert_called_once()
        call_args = self.socket_manager._send_packet.call_args[0]
        self.assertEqual(use_packet, call_args[0])
        self.assertEqual(connection_id, call_args[1])
        self.assertEqual(send_time, call_args[2])

        self.socket_manager.close_all_channels()
        time.sleep(0.1)
        self.socket_manager._safe_put(connection_id, (send_time, use_packet))
        self.assertEqual(use_packet.status, Packet.STATUS_FAIL)


class TestSocketManagerMessageHandling(unittest.TestCase):
    '''Test sending messages to the world and then to each of two agents,
    along with failure cases for each
    '''

    def on_alive(self, packet):
        self.alive_packet = packet
        self.socket_manager.open_channel(
            packet.sender_id, packet.assignment_id)

    def on_message(self, packet):
        self.message_packet = packet

    def on_worker_death(self, worker_id, assignment_id):
        self.dead_worker_id = worker_id
        self.dead_assignment_id = assignment_id

    def on_server_death(self):
        self.server_died = True

    def assertEqualBy(self, val_func, val, max_time):
        start_time = time.time()
        while val_func() != val:
            assert time.time() - start_time < max_time, \
                "Value was not attained in specified time"
            time.sleep(0.1)

    def setUp(self):
        self.fake_socket = MockSocket()
        time.sleep(0.3)
        self.agent1 = MockAgent(TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1,
                                TEST_WORKER_ID_1, TASK_GROUP_ID_1)
        self.agent2 = MockAgent(TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2,
                                TEST_WORKER_ID_2, TASK_GROUP_ID_1)
        self.alive_packet = None
        self.message_packet = None
        self.dead_worker_id = None
        self.dead_assignment_id = None
        self.server_died = False

        self.socket_manager = SocketManager(
            'https://127.0.0.1', 3030, self.on_alive, self.on_message,
            self.on_worker_death, TASK_GROUP_ID_1, 1, self.on_server_death)

    def tearDown(self):
        self.socket_manager.shutdown()
        self.fake_socket.close()

    def test_alive_send_and_disconnect(self):
        acked_packet = None
        incoming_hb = None
        message_packet = None
        hb_count = 0

        def on_ack(*args):
            nonlocal acked_packet
            acked_packet = args[0]

        def on_hb(*args):
            nonlocal incoming_hb, hb_count
            incoming_hb = args[0]
            hb_count += 1

        def on_msg(*args):
            nonlocal message_packet
            message_packet = args[0]

        self.agent1.register_to_socket(self.fake_socket, on_ack, on_hb, on_msg)
        self.assertIsNone(acked_packet)
        self.assertIsNone(incoming_hb)
        self.assertIsNone(message_packet)
        self.assertEqual(hb_count, 0)

        # Assert alive is registered
        alive_id = self.agent1.send_alive()
        self.assertEqualBy(lambda: acked_packet is None, False, 8)
        self.assertIsNone(incoming_hb)
        self.assertIsNone(message_packet)
        self.assertIsNone(self.message_packet)
        self.assertEqualBy(lambda: self.alive_packet is None, False, 8)
        self.assertEqual(self.alive_packet.id, alive_id)
        self.assertEqual(acked_packet.id, alive_id, 'Alive was not acked')
        acked_packet = None

        # assert sending heartbeats actually works, and that heartbeats don't
        # get acked
        self.agent1.send_heartbeat()
        self.assertEqualBy(lambda: incoming_hb is None, False, 8)
        self.assertIsNone(acked_packet)
        self.assertGreater(hb_count, 0)

        # Test message send from agent
        test_message_text_1 = 'test_message_text_1'
        msg_id = self.agent1.send_message(test_message_text_1)
        self.assertEqualBy(lambda: self.message_packet is None, False, 8)
        self.assertEqualBy(lambda: acked_packet is None, False, 8)
        self.assertEqual(self.message_packet.id, acked_packet.id)
        self.assertEqual(self.message_packet.id, msg_id)
        self.assertEqual(self.message_packet.data['text'], test_message_text_1)

        # Test message send to agent
        manager_message_id = 'message_id_from_manager'
        test_message_text_2 = 'test_message_text_2'
        message_send_packet = Packet(
            manager_message_id, Packet.TYPE_MESSAGE,
            self.socket_manager.get_my_sender_id(), TEST_WORKER_ID_1,
            TEST_ASSIGNMENT_ID_1, test_message_text_2, 't2')
        self.socket_manager.queue_packet(message_send_packet)
        self.assertEqualBy(lambda: message_packet is None, False, 8)
        self.assertEqual(message_packet.id, manager_message_id)
        self.assertEqual(message_packet.data, test_message_text_2)
        self.assertIn(manager_message_id, self.socket_manager.packet_map)
        self.assertEqualBy(
            lambda: self.socket_manager.packet_map[manager_message_id].status,
            Packet.STATUS_ACK,
            6,
        )

        # Test agent disconnect
        self.agent1.always_beat = False
        self.assertEqualBy(lambda: self.dead_worker_id, TEST_WORKER_ID_1, 8)
        self.assertEqual(self.dead_assignment_id, TEST_ASSIGNMENT_ID_1)
        self.assertGreater(hb_count, 1)

    def test_failed_ack_resend(self):
        '''Ensures when a message from the manager is dropped, it gets
        retried until it works as long as there hasn't been a disconnect
        '''
        acked_packet = None
        incoming_hb = None
        message_packet = None
        hb_count = 0

        def on_ack(*args):
            nonlocal acked_packet
            acked_packet = args[0]

        def on_hb(*args):
            nonlocal incoming_hb, hb_count
            incoming_hb = args[0]
            hb_count += 1

        def on_msg(*args):
            nonlocal message_packet
            message_packet = args[0]

        self.agent1.register_to_socket(self.fake_socket, on_ack, on_hb, on_msg)
        self.assertIsNone(acked_packet)
        self.assertIsNone(incoming_hb)
        self.assertIsNone(message_packet)
        self.assertEqual(hb_count, 0)

        # Assert alive is registered
        alive_id = self.agent1.send_alive()
        self.assertEqualBy(lambda: acked_packet is None, False, 8)
        self.assertIsNone(incoming_hb)
        self.assertIsNone(message_packet)
        self.assertIsNone(self.message_packet)
        self.assertEqualBy(lambda: self.alive_packet is None, False, 8)
        self.assertEqual(self.alive_packet.id, alive_id)
        self.assertEqual(acked_packet.id, alive_id, 'Alive was not acked')
        acked_packet = None

        # assert sending heartbeats actually works, and that heartbeats don't
        # get acked
        self.agent1.send_heartbeat()
        self.assertEqualBy(lambda: incoming_hb is None, False, 8)
        self.assertIsNone(acked_packet)
        self.assertGreater(hb_count, 0)

        # Test message send to agent
        manager_message_id = 'message_id_from_manager'
        test_message_text_2 = 'test_message_text_2'
        self.agent1.send_acks = False
        message_send_packet = Packet(
            manager_message_id, Packet.TYPE_MESSAGE,
            self.socket_manager.get_my_sender_id(), TEST_WORKER_ID_1,
            TEST_ASSIGNMENT_ID_1, test_message_text_2, 't2')
        self.socket_manager.queue_packet(message_send_packet)
        self.assertEqualBy(lambda: message_packet is None, False, 8)
        self.assertEqual(message_packet.id, manager_message_id)
        self.assertEqual(message_packet.data, test_message_text_2)
        self.assertIn(manager_message_id, self.socket_manager.packet_map)
        self.assertNotEqual(
            self.socket_manager.packet_map[manager_message_id].status,
            Packet.STATUS_ACK,
        )
        message_packet = None
        self.agent1.send_acks = True
        self.assertEqualBy(lambda: message_packet is None, False, 8)
        self.assertEqual(message_packet.id, manager_message_id)
        self.assertEqual(message_packet.data, test_message_text_2)
        self.assertIn(manager_message_id, self.socket_manager.packet_map)
        self.assertEqualBy(
            lambda: self.socket_manager.packet_map[manager_message_id].status,
            Packet.STATUS_ACK,
            6,
        )

    def test_one_agent_disconnect_other_alive(self):
        acked_packet = None
        incoming_hb = None
        message_packet = None
        hb_count = 0

        def on_ack(*args):
            nonlocal acked_packet
            acked_packet = args[0]

        def on_hb(*args):
            nonlocal incoming_hb, hb_count
            incoming_hb = args[0]
            hb_count += 1

        def on_msg(*args):
            nonlocal message_packet
            message_packet = args[0]

        self.agent1.register_to_socket(self.fake_socket, on_ack, on_hb, on_msg)
        self.agent2.register_to_socket(self.fake_socket, on_ack, on_hb, on_msg)
        self.assertIsNone(acked_packet)
        self.assertIsNone(incoming_hb)
        self.assertIsNone(message_packet)
        self.assertEqual(hb_count, 0)

        # Assert alive is registered
        self.agent1.send_alive()
        self.agent2.send_alive()
        self.assertEqualBy(lambda: acked_packet is None, False, 8)
        self.assertIsNone(incoming_hb)
        self.assertIsNone(message_packet)

        # Start sending heartbeats
        self.agent1.send_heartbeat()
        self.agent2.send_heartbeat()

        # Kill second agent
        self.agent2.always_beat = False
        self.assertEqualBy(lambda: self.dead_worker_id, TEST_WORKER_ID_2, 8)
        self.assertEqual(self.dead_assignment_id, TEST_ASSIGNMENT_ID_2)

        # Run rest of tests

        # Test message send from agent
        test_message_text_1 = 'test_message_text_1'
        msg_id = self.agent1.send_message(test_message_text_1)
        self.assertEqualBy(lambda: self.message_packet is None, False, 8)
        self.assertEqualBy(lambda: acked_packet is None, False, 8)
        self.assertEqual(self.message_packet.id, acked_packet.id)
        self.assertEqual(self.message_packet.id, msg_id)
        self.assertEqual(self.message_packet.data['text'], test_message_text_1)

        # Test message send to agent
        manager_message_id = 'message_id_from_manager'
        test_message_text_2 = 'test_message_text_2'
        message_send_packet = Packet(
            manager_message_id, Packet.TYPE_MESSAGE,
            self.socket_manager.get_my_sender_id(), TEST_WORKER_ID_1,
            TEST_ASSIGNMENT_ID_1, test_message_text_2, 't2')
        self.socket_manager.queue_packet(message_send_packet)
        self.assertEqualBy(lambda: message_packet is None, False, 8)
        self.assertEqual(message_packet.id, manager_message_id)
        self.assertEqual(message_packet.data, test_message_text_2)
        self.assertIn(manager_message_id, self.socket_manager.packet_map)
        self.assertEqualBy(
            lambda: self.socket_manager.packet_map[manager_message_id].status,
            Packet.STATUS_ACK,
            6,
        )

        # Test agent disconnect
        self.agent1.always_beat = False
        self.assertEqualBy(lambda: self.dead_worker_id, TEST_WORKER_ID_1, 8)
        self.assertEqual(self.dead_assignment_id, TEST_ASSIGNMENT_ID_1)


if __name__ == '__main__':
    unittest.main(buffer=True)
