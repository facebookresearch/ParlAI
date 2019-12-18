#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.agents import AssignState

import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
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

MESSAGE_TYPE = data_model.MESSAGE_TYPE_ACT
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

SocketManager.DEF_MISSED_PONGS = 3
SocketManager.DEF_DEAD_TIME = 0.6

shared_utils.THREAD_SHORT_SLEEP = 0.05
shared_utils.THREAD_MEDIUM_SLEEP = 0.15


class TestPacket(unittest.TestCase):
    """
    Various unit tests for the AssignState class.
    """

    ID = 'ID'
    SENDER_ID = 'SENDER_ID'
    RECEIVER_ID = 'RECEIVER_ID'
    ASSIGNMENT_ID = 'ASSIGNMENT_ID'
    DATA = 'DATA'
    CONVERSATION_ID = 'CONVERSATION_ID'
    ACK_FUNCTION = 'ACK_FUNCTION'

    def setUp(self):
        self.packet_1 = Packet(
            self.ID,
            data_model.MESSAGE_BATCH,
            self.SENDER_ID,
            self.RECEIVER_ID,
            self.ASSIGNMENT_ID,
            self.DATA,
            conversation_id=self.CONVERSATION_ID,
            ack_func=self.ACK_FUNCTION,
        )
        self.packet_2 = Packet(
            self.ID,
            data_model.SNS_MESSAGE,
            self.SENDER_ID,
            self.RECEIVER_ID,
            self.ASSIGNMENT_ID,
            self.DATA,
        )
        self.packet_3 = Packet(
            self.ID,
            data_model.AGENT_ALIVE,
            self.SENDER_ID,
            self.RECEIVER_ID,
            self.ASSIGNMENT_ID,
            self.DATA,
        )

    def tearDown(self):
        pass

    def test_packet_init(self):
        """
        Test proper initialization of packet fields.
        """
        self.assertEqual(self.packet_1.id, self.ID)
        self.assertEqual(self.packet_1.type, data_model.MESSAGE_BATCH)
        self.assertEqual(self.packet_1.sender_id, self.SENDER_ID)
        self.assertEqual(self.packet_1.receiver_id, self.RECEIVER_ID)
        self.assertEqual(self.packet_1.assignment_id, self.ASSIGNMENT_ID)
        self.assertEqual(self.packet_1.data, self.DATA)
        self.assertEqual(self.packet_1.conversation_id, self.CONVERSATION_ID)
        self.assertEqual(self.packet_1.ack_func, self.ACK_FUNCTION)
        self.assertEqual(self.packet_1.status, Packet.STATUS_INIT)
        self.assertEqual(self.packet_2.id, self.ID)
        self.assertEqual(self.packet_2.type, data_model.SNS_MESSAGE)
        self.assertEqual(self.packet_2.sender_id, self.SENDER_ID)
        self.assertEqual(self.packet_2.receiver_id, self.RECEIVER_ID)
        self.assertEqual(self.packet_2.assignment_id, self.ASSIGNMENT_ID)
        self.assertEqual(self.packet_2.data, self.DATA)
        self.assertIsNone(self.packet_2.conversation_id)
        self.assertIsNone(self.packet_2.ack_func)
        self.assertEqual(self.packet_2.status, Packet.STATUS_INIT)
        self.assertEqual(self.packet_3.id, self.ID)
        self.assertEqual(self.packet_3.type, data_model.AGENT_ALIVE)
        self.assertEqual(self.packet_3.sender_id, self.SENDER_ID)
        self.assertEqual(self.packet_3.receiver_id, self.RECEIVER_ID)
        self.assertEqual(self.packet_3.assignment_id, self.ASSIGNMENT_ID)
        self.assertEqual(self.packet_3.data, self.DATA)
        self.assertIsNone(self.packet_3.conversation_id)
        self.assertIsNone(self.packet_3.ack_func)
        self.assertEqual(self.packet_3.status, Packet.STATUS_INIT)

    def test_dict_conversion(self):
        """
        Ensure packets can be converted to and from a representative dict.
        """
        converted_packet = Packet.from_dict(self.packet_1.as_dict())
        self.assertEqual(self.packet_1.id, converted_packet.id)
        self.assertEqual(self.packet_1.type, converted_packet.type)
        self.assertEqual(self.packet_1.sender_id, converted_packet.sender_id)
        self.assertEqual(self.packet_1.receiver_id, converted_packet.receiver_id)
        self.assertEqual(self.packet_1.assignment_id, converted_packet.assignment_id)
        self.assertEqual(self.packet_1.data, converted_packet.data)
        self.assertEqual(
            self.packet_1.conversation_id, converted_packet.conversation_id
        )

        packet_dict = self.packet_1.as_dict()
        self.assertDictEqual(packet_dict, Packet.from_dict(packet_dict).as_dict())

    def test_connection_ids(self):
        """
        Ensure that connection ids are reported as we expect them.
        """
        sender_conn_id = '{}_{}'.format(self.SENDER_ID, self.ASSIGNMENT_ID)
        receiver_conn_id = '{}_{}'.format(self.RECEIVER_ID, self.ASSIGNMENT_ID)
        self.assertEqual(self.packet_1.get_sender_connection_id(), sender_conn_id)
        self.assertEqual(self.packet_1.get_receiver_connection_id(), receiver_conn_id)

    def test_packet_conversions(self):
        """
        Ensure that packet copies and acts are produced properly.
        """
        # Copy important packet
        message_packet_copy = self.packet_1.new_copy()
        self.assertNotEqual(message_packet_copy.id, self.ID)
        self.assertNotEqual(message_packet_copy, self.packet_1)
        self.assertEqual(message_packet_copy.type, self.packet_1.type)
        self.assertEqual(message_packet_copy.sender_id, self.packet_1.sender_id)
        self.assertEqual(message_packet_copy.receiver_id, self.packet_1.receiver_id)
        self.assertEqual(message_packet_copy.assignment_id, self.packet_1.assignment_id)
        self.assertEqual(message_packet_copy.data, self.packet_1.data)
        self.assertEqual(
            message_packet_copy.conversation_id, self.packet_1.conversation_id
        )
        self.assertIsNone(message_packet_copy.ack_func)
        self.assertEqual(message_packet_copy.status, Packet.STATUS_INIT)

        # Copy non-important packet
        hb_packet_copy = self.packet_2.new_copy()
        self.assertNotEqual(hb_packet_copy.id, self.ID)
        self.assertNotEqual(hb_packet_copy, self.packet_2)
        self.assertEqual(hb_packet_copy.type, self.packet_2.type)
        self.assertEqual(hb_packet_copy.sender_id, self.packet_2.sender_id)
        self.assertEqual(hb_packet_copy.receiver_id, self.packet_2.receiver_id)
        self.assertEqual(hb_packet_copy.assignment_id, self.packet_2.assignment_id)
        self.assertEqual(hb_packet_copy.data, self.packet_2.data)
        self.assertEqual(hb_packet_copy.conversation_id, self.packet_2.conversation_id)
        self.assertIsNone(hb_packet_copy.ack_func)
        self.assertEqual(hb_packet_copy.status, Packet.STATUS_INIT)

    def test_packet_modifications(self):
        """
        Ensure that packet copies and acts are produced properly.
        """
        # All operations return the packet
        self.assertEqual(self.packet_1.swap_sender(), self.packet_1)
        self.assertEqual(
            self.packet_1.set_type(data_model.MESSAGE_BATCH), self.packet_1
        )
        self.assertEqual(self.packet_1.set_data(None), self.packet_1)

        # Ensure all of the operations worked
        self.assertEqual(self.packet_1.sender_id, self.RECEIVER_ID)
        self.assertEqual(self.packet_1.receiver_id, self.SENDER_ID)
        self.assertEqual(self.packet_1.type, data_model.MESSAGE_BATCH)
        self.assertIsNone(self.packet_1.data)


class MockSocket:
    def __init__(self):
        self.last_messages = {}
        self.connected = False
        self.disconnected = False
        self.closed = False
        self.ws = None
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
            elif packet_dict['type'] == data_model.WORLD_PING:
                pong = packet_dict['content'].copy()
                pong['type'] = 'pong'
                self.ws.send_message(
                    client,
                    json.dumps({'type': data_model.SERVER_PONG, 'content': pong}),
                )
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
        self.ready = False
        self.wants_to_send = False

    def send_packet(self, packet):
        def callback(*args):
            pass

        event_name = data_model.MESSAGE_BATCH
        self.ws.send(json.dumps({'type': event_name, 'content': packet.as_dict()}))

    def register_to_socket(self, ws, on_msg):
        handler = self.make_packet_handler(on_msg)
        self.ws = ws
        self.ws.handlers[self.worker_id] = handler

    def make_packet_handler(self, on_msg):
        """
        A packet handler.
        """

        def handler_mock(pkt):
            if pkt['type'] == data_model.WORLD_MESSAGE:
                packet = Packet.from_dict(pkt)
                on_msg(packet)
            elif pkt['type'] == data_model.MESSAGE_BATCH:
                packet = Packet.from_dict(pkt)
                on_msg(packet)
            elif pkt['type'] == data_model.AGENT_ALIVE:
                raise Exception('Invalid alive packet {}'.format(pkt))
            else:
                raise Exception(
                    'Invalid Packet type {} received in {}'.format(pkt['type'], pkt)
                )

        return handler_mock

    def build_and_send_packet(self, packet_type, data):
        msg_id = str(uuid.uuid4())
        msg = {
            'id': msg_id,
            'type': packet_type,
            'sender_id': self.worker_id,
            'assignment_id': self.assignment_id,
            'conversation_id': self.conversation_id,
            'receiver_id': '[World_' + self.task_group_id + ']',
            'data': data,
        }

        if packet_type == data_model.MESSAGE_BATCH:
            msg['data'] = {
                'messages': [
                    {
                        'id': msg_id,
                        'type': packet_type,
                        'sender_id': self.worker_id,
                        'assignment_id': self.assignment_id,
                        'conversation_id': self.conversation_id,
                        'receiver_id': '[World_' + self.task_group_id + ']',
                        'data': data,
                    }
                ]
            }
        self.ws.send(json.dumps({'type': packet_type, 'content': msg}))
        return msg['id']

    def send_message(self, text):
        data = {
            'text': text,
            'id': self.id,
            'message_id': str(uuid.uuid4()),
            'episode_done': False,
        }

        self.wants_to_send = False
        return self.build_and_send_packet(data_model.MESSAGE_BATCH, data)

    def send_disconnect(self):
        data = {
            'hit_id': self.hit_id,
            'assignment_id': self.assignment_id,
            'worker_id': self.worker_id,
            'conversation_id': self.conversation_id,
            'connection_id': '{}_{}'.format(self.worker_id, self.assignment_id),
        }

        return self.build_and_send_packet(data_model.AGENT_DISCONNECT, data)

    def send_alive(self):
        data = {
            'hit_id': self.hit_id,
            'assignment_id': self.assignment_id,
            'worker_id': self.worker_id,
            'conversation_id': self.conversation_id,
        }
        return self.build_and_send_packet(data_model.AGENT_ALIVE, data)


class TestSocketManagerSetupAndFunctions(unittest.TestCase):
    """
    Unit/integration tests for starting up a socket.
    """

    def setUp(self):
        self.fake_socket = MockSocket()
        time.sleep(1)

    def tearDown(self):
        self.fake_socket.close()

    def test_init_and_reg_shutdown(self):
        """
        Test initialization of a socket manager.
        """
        self.assertFalse(self.fake_socket.connected)

        # Callbacks should never trigger during proper setup and shutdown
        nop_called = False

        def nop(*args):
            nonlocal nop_called  # noqa 999 we don't support py2
            nop_called = True

        socket_manager = SocketManager(
            'https://127.0.0.1',
            self.fake_socket.port,
            nop,
            nop,
            nop,
            TASK_GROUP_ID_1,
            0.3,
            nop,
        )
        self.assertTrue(self.fake_socket.connected)
        self.assertFalse(nop_called)

        # Test shutdown
        self.assertFalse(self.fake_socket.disconnected)
        self.assertFalse(socket_manager.is_shutdown)
        self.assertTrue(socket_manager.alive)
        socket_manager.shutdown()
        time.sleep(0.3)
        self.assertTrue(self.fake_socket.disconnected)
        self.assertTrue(socket_manager.is_shutdown)
        self.assertFalse(nop_called)

    def assertEqualBy(self, val_func, val, max_time):
        start_time = time.time()
        while val_func() != val:
            assert time.time() - start_time < max_time, (
                "Value was not attained in specified time, was {} rather "
                "than {}".format(val_func(), val)
            )
            time.sleep(0.1)

    def test_init_and_socket_shutdown(self):
        """
        Test initialization of a socket manager with a failed shutdown.
        """
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

        socket_manager = SocketManager(
            'https://127.0.0.1',
            self.fake_socket.port,
            nop,
            nop,
            nop,
            TASK_GROUP_ID_1,
            0.4,
            server_death,
        )
        self.assertTrue(self.fake_socket.connected)
        self.assertFalse(nop_called)
        self.assertFalse(server_death_called)

        # Test shutdown
        self.assertFalse(self.fake_socket.disconnected)
        self.assertFalse(socket_manager.is_shutdown)
        self.assertTrue(socket_manager.alive)
        self.fake_socket.close()
        self.assertEqualBy(lambda: socket_manager.alive, False, 8)
        self.assertEqualBy(lambda: server_death_called, True, 20)
        self.assertFalse(nop_called)
        socket_manager.shutdown()

    def test_init_and_socket_shutdown_then_restart(self):
        """
        Test restoring connection to a socket.
        """
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

        socket_manager = SocketManager(
            'https://127.0.0.1',
            self.fake_socket.port,
            nop,
            nop,
            nop,
            TASK_GROUP_ID_1,
            0.4,
            server_death,
        )
        self.assertTrue(self.fake_socket.connected)
        self.assertFalse(nop_called)
        self.assertFalse(server_death_called)

        # Test shutdown
        self.assertFalse(self.fake_socket.disconnected)
        self.assertFalse(socket_manager.is_shutdown)
        self.assertTrue(socket_manager.alive)
        self.fake_socket.close()
        self.assertEqualBy(lambda: socket_manager.alive, False, 8)
        self.assertFalse(socket_manager.alive)
        self.fake_socket = MockSocket()
        self.assertEqualBy(lambda: socket_manager.alive, True, 4)
        self.assertFalse(nop_called)
        self.assertFalse(server_death_called)
        socket_manager.shutdown()

    def test_init_world_dead(self):
        """
        Test initialization of a socket manager with a failed startup.
        """
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
            socket_manager = SocketManager(
                'https://127.0.0.1',
                self.fake_socket.port,
                nop,
                nop,
                nop,
                TASK_GROUP_ID_1,
                0.4,
                server_death,
            )
            self.assertIsNone(socket_manager)

        self.assertFalse(nop_called)
        self.assertTrue(server_death_called)


class TestSocketManagerRoutingFunctionality(unittest.TestCase):

    ID = 'ID'
    SENDER_ID = 'SENDER_ID'
    ASSIGNMENT_ID = 'ASSIGNMENT_ID'
    DATA = 'DATA'
    CONVERSATION_ID = 'CONVERSATION_ID'
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
        self.AGENT_ALIVE_PACKET = Packet(
            MESSAGE_ID_1,
            data_model.AGENT_ALIVE,
            self.SENDER_ID,
            self.WORLD_ID,
            self.ASSIGNMENT_ID,
            self.DATA,
            self.CONVERSATION_ID,
        )

        self.MESSAGE_SEND_PACKET_1 = Packet(
            MESSAGE_ID_2,
            data_model.WORLD_MESSAGE,
            self.WORLD_ID,
            self.SENDER_ID,
            self.ASSIGNMENT_ID,
            self.DATA,
            self.CONVERSATION_ID,
        )

        self.MESSAGE_SEND_PACKET_2 = Packet(
            MESSAGE_ID_3,
            data_model.MESSAGE_BATCH,
            self.WORLD_ID,
            self.SENDER_ID,
            self.ASSIGNMENT_ID,
            self.DATA,
            self.CONVERSATION_ID,
        )

        self.MESSAGE_SEND_PACKET_3 = Packet(
            MESSAGE_ID_4,
            data_model.MESSAGE_BATCH,
            self.WORLD_ID,
            self.SENDER_ID,
            self.ASSIGNMENT_ID,
            self.DATA,
            self.CONVERSATION_ID,
        )

        self.fake_socket = MockSocket()
        time.sleep(0.3)
        self.alive_packet = None
        self.message_packet = None
        self.dead_worker_id = None
        self.dead_assignment_id = None
        self.server_died = False

        self.socket_manager = SocketManager(
            'https://127.0.0.1',
            self.fake_socket.port,
            self.on_alive,
            self.on_message,
            self.on_worker_death,
            TASK_GROUP_ID_1,
            1,
            self.on_server_death,
        )

    def tearDown(self):
        self.socket_manager.shutdown()
        self.fake_socket.close()

    def test_init_state(self):
        """
        Ensure all of the initial state of the socket_manager is ready.
        """
        self.assertEqual(self.socket_manager.server_url, 'https://127.0.0.1')
        self.assertEqual(self.socket_manager.port, self.fake_socket.port)
        self.assertEqual(self.socket_manager.alive_callback, self.on_alive)
        self.assertEqual(self.socket_manager.message_callback, self.on_message)
        self.assertEqual(self.socket_manager.socket_dead_callback, self.on_worker_death)
        self.assertEqual(self.socket_manager.task_group_id, TASK_GROUP_ID_1)
        self.assertEqual(
            self.socket_manager.missed_pongs, 1 + (1 / SocketManager.PING_RATE)
        )
        self.assertIsNotNone(self.socket_manager.ws)
        self.assertTrue(self.socket_manager.keep_running)
        self.assertIsNotNone(self.socket_manager.listen_thread)
        self.assertSetEqual(self.socket_manager.open_channels, set())
        self.assertDictEqual(self.socket_manager.packet_map, {})
        self.assertTrue(self.socket_manager.alive)
        self.assertFalse(self.socket_manager.is_shutdown)
        self.assertEqual(self.socket_manager.get_my_sender_id(), self.WORLD_ID)

    def _send_packet_in_background(self, packet, send_time):
        """
        creates a thread to handle waiting for a packet send.
        """

        def do_send():
            self.socket_manager._send_packet(packet, send_time)
            self.sent = True

        send_thread = threading.Thread(target=do_send, daemon=True)
        send_thread.start()
        time.sleep(0.02)

    def test_packet_send(self):
        """
        Checks to see if packets are working.
        """
        self.socket_manager._safe_send = mock.MagicMock()
        self.sent = False

        # Test a blocking acknowledged packet
        send_time = time.time()
        self.assertEqual(self.MESSAGE_SEND_PACKET_2.status, Packet.STATUS_INIT)
        self._send_packet_in_background(self.MESSAGE_SEND_PACKET_2, send_time)
        self.assertEqual(self.MESSAGE_SEND_PACKET_2.status, Packet.STATUS_SENT)
        self.socket_manager._safe_send.assert_called_once()
        self.assertTrue(self.sent)

        used_packet_json = self.socket_manager._safe_send.call_args[0][0]
        used_packet_dict = json.loads(used_packet_json)
        self.assertEqual(used_packet_dict['type'], data_model.MESSAGE_BATCH)
        self.assertDictEqual(
            used_packet_dict['content'], self.MESSAGE_SEND_PACKET_2.as_dict()
        )

    def test_simple_packet_channel_management(self):
        """
        Ensure that channels are created, managed, and then removed as expected.
        """
        use_packet = self.MESSAGE_SEND_PACKET_1
        worker_id = use_packet.receiver_id
        assignment_id = use_packet.assignment_id

        # Open a channel and assert it is there
        self.socket_manager.open_channel(worker_id, assignment_id)
        time.sleep(0.1)
        connection_id = use_packet.get_receiver_connection_id()

        self.assertIn(connection_id, self.socket_manager.open_channels)
        self.assertTrue(self.socket_manager.socket_is_open(connection_id))
        self.assertFalse(self.socket_manager.socket_is_open(FAKE_ID))

        # Send a packet to an open socket, ensure it got queued
        resp = self.socket_manager.queue_packet(use_packet)
        self.assertIn(use_packet.id, self.socket_manager.packet_map)
        self.assertTrue(resp)

        # Assert we can get the status of a packet in the map, but not
        # existing doesn't throw an error
        self.assertEqual(
            self.socket_manager.get_status(use_packet.id), use_packet.status
        )
        self.assertEqual(self.socket_manager.get_status(FAKE_ID), Packet.STATUS_NONE)

        # Assert that closing a thread does the correct cleanup work
        self.socket_manager.close_channel(connection_id)
        time.sleep(0.2)
        self.assertNotIn(connection_id, self.socket_manager.open_channels)
        self.assertNotIn(use_packet.id, self.socket_manager.packet_map)

        # Assert that opening multiple and closing them is possible
        self.socket_manager.open_channel(worker_id, assignment_id)
        self.socket_manager.open_channel(worker_id + '2', assignment_id)
        time.sleep(0.1)
        self.assertEqual(len(self.socket_manager.open_channels), 2)
        self.socket_manager.close_all_channels()
        time.sleep(0.1)
        self.assertEqual(len(self.socket_manager.open_channels), 0)


class TestSocketManagerMessageHandling(unittest.TestCase):
    """
    Test sending messages to the world and then to each of two agents, along with
    failure cases for each.
    """

    def on_alive(self, packet):
        self.alive_packet = packet
        self.socket_manager.open_channel(packet.sender_id, packet.assignment_id)

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
            assert (
                time.time() - start_time < max_time
            ), "Value was not attained in specified time"
            time.sleep(0.1)

    def setUp(self):
        self.fake_socket = MockSocket()
        time.sleep(0.3)
        self.agent1 = MockAgent(
            TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1, TASK_GROUP_ID_1
        )
        self.agent2 = MockAgent(
            TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2, TASK_GROUP_ID_1
        )
        self.alive_packet = None
        self.message_packet = None
        self.dead_worker_id = None
        self.dead_assignment_id = None
        self.server_died = False

        self.socket_manager = SocketManager(
            'https://127.0.0.1',
            3030,
            self.on_alive,
            self.on_message,
            self.on_worker_death,
            TASK_GROUP_ID_1,
            1,
            self.on_server_death,
        )

    def tearDown(self):
        self.socket_manager.shutdown()
        self.fake_socket.close()

    def test_alive_send_and_disconnect(self):
        message_packet = None

        def on_msg(*args):
            nonlocal message_packet
            message_packet = args[0]

        self.agent1.register_to_socket(self.fake_socket, on_msg)
        self.assertIsNone(message_packet)

        # Assert alive is registered
        alive_id = self.agent1.send_alive()
        self.assertIsNone(message_packet)
        self.assertIsNone(self.message_packet)
        self.assertEqualBy(lambda: self.alive_packet is None, False, 8)
        self.assertEqual(self.alive_packet.id, alive_id)

        # Test message send from agent
        test_message_text_1 = 'test_message_text_1'
        msg_id = self.agent1.send_message(test_message_text_1)
        self.assertEqualBy(lambda: self.message_packet is None, False, 8)
        self.assertEqual(self.message_packet.id, msg_id)
        self.assertEqual(self.message_packet.data['text'], test_message_text_1)

        # Test message send to agent
        manager_message_id = 'message_id_from_manager'
        test_message_text_2 = 'test_message_text_2'
        message_send_packet = Packet(
            manager_message_id,
            data_model.MESSAGE_BATCH,
            self.socket_manager.get_my_sender_id(),
            TEST_WORKER_ID_1,
            TEST_ASSIGNMENT_ID_1,
            test_message_text_2,
            't2',
        )
        self.socket_manager.queue_packet(message_send_packet)
        self.assertEqualBy(lambda: message_packet is None, False, 8)
        self.assertEqual(message_packet.id, manager_message_id)
        self.assertEqual(message_packet.data, test_message_text_2)
        self.assertIn(manager_message_id, self.socket_manager.packet_map)

        # Test agent disconnect
        self.agent1.send_disconnect()
        self.assertEqualBy(lambda: self.dead_worker_id, TEST_WORKER_ID_1, 8)
        self.assertEqual(self.dead_assignment_id, TEST_ASSIGNMENT_ID_1)

    def test_one_agent_disconnect_other_alive(self):
        message_packet = None

        def on_msg(*args):
            nonlocal message_packet
            message_packet = args[0]

        self.agent1.register_to_socket(self.fake_socket, on_msg)
        self.agent2.register_to_socket(self.fake_socket, on_msg)
        self.assertIsNone(message_packet)

        # Assert alive is registered
        self.agent1.send_alive()
        self.agent2.send_alive()
        self.assertIsNone(message_packet)

        # Kill second agent
        self.agent2.send_disconnect()
        self.assertEqualBy(lambda: self.dead_worker_id, TEST_WORKER_ID_2, 8)
        self.assertEqual(self.dead_assignment_id, TEST_ASSIGNMENT_ID_2)

        # Run rest of tests

        # Test message send from agent
        test_message_text_1 = 'test_message_text_1'
        msg_id = self.agent1.send_message(test_message_text_1)
        self.assertEqualBy(lambda: self.message_packet is None, False, 8)
        self.assertEqual(self.message_packet.id, msg_id)
        self.assertEqual(self.message_packet.data['text'], test_message_text_1)

        # Test message send to agent
        manager_message_id = 'message_id_from_manager'
        test_message_text_2 = 'test_message_text_2'
        message_send_packet = Packet(
            manager_message_id,
            data_model.WORLD_MESSAGE,
            self.socket_manager.get_my_sender_id(),
            TEST_WORKER_ID_1,
            TEST_ASSIGNMENT_ID_1,
            test_message_text_2,
            't2',
        )
        self.socket_manager.queue_packet(message_send_packet)
        self.assertEqualBy(lambda: message_packet is None, False, 8)
        self.assertEqual(message_packet.id, manager_message_id)
        self.assertEqual(message_packet.data, test_message_text_2)
        self.assertIn(manager_message_id, self.socket_manager.packet_map)

        # Test agent disconnect
        self.agent1.send_disconnect()
        self.assertEqualBy(lambda: self.dead_worker_id, TEST_WORKER_ID_1, 8)
        self.assertEqual(self.dead_assignment_id, TEST_ASSIGNMENT_ID_1)


if __name__ == '__main__':
    unittest.main(buffer=True)
