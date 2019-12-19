#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import time
import uuid
import os
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.mturk.core.dev.agents import AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser

import parlai.mturk.core.dev.mturk_manager as MTurkManagerFile
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
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

SocketManager.DEF_MISSED_PONGS = 1
SocketManager.DEF_DEAD_TIME = 0.4

shared_utils.THREAD_SHORT_SLEEP = 0.05
shared_utils.THREAD_MEDIUM_SLEEP = 0.15

MTurkManagerFile.WORLD_START_TIMEOUT = 2


TOPIC_ARN = 'topic_arn'
QUALIFICATION_ID = 'qualification_id'
HIT_TYPE_ID = 'hit_type_id'
MTURK_PAGE_URL = 'mturk_page_url'
FAKE_HIT_ID = 'fake_hit_id'


class TestMTurkWorld(MTurkTaskWorld):
    def __init__(self, workers, use_episode_done):
        self.workers = workers

        def episode_done():
            return use_episode_done()

        self.episode_done = episode_done

    def parley(self):
        for worker in self.workers:
            worker.assert_connected()
        time.sleep(0.5)

    def shutdown(self):
        for worker in self.workers:
            worker.shutdown()


class TestMTurkOnboardWorld(MTurkOnboardWorld):
    def __init__(self, mturk_agent, use_episode_done):
        self.mturk_agent = mturk_agent

        def episode_done():
            return use_episode_done()

        self.episode_done = episode_done

    def parley(self):
        self.mturk_agent.assert_connected()
        time.sleep(0.5)


def assert_equal_by(val_func, val, max_time):
    start_time = time.time()
    while val_func() != val:
        assert (
            time.time() - start_time < max_time
        ), "Value was not attained in specified time, last {}".format(val_func())
        time.sleep(0.1)


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
        self.message_packet = []

    def send_packet(self, packet):
        def callback(*args):
            pass

        event_name = data_model.MESSAGE_BATCH
        self.ws.send(json.dumps({'type': event_name, 'content': packet.as_dict()}))

    def register_to_socket(self, ws, on_msg=None):
        if on_msg is None:

            def on_msg(packet):
                self.message_packet.append(packet)
                if packet.type == data_model.AGENT_STATE_CHANGE:
                    if 'conversation_id' in packet.data:
                        self.conversation_id = packet.data['conversation_id']
                    if 'agent_id' in packet.data:
                        self.id = packet.data['agent_id']

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
            elif pkt['type'] == data_model.AGENT_STATE_CHANGE:
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
        self.opt['frontend_version'] = 1
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
        self.mturk_manager.set_get_onboard_world(self.get_onboard_world)
        self.mturk_manager.create_hits()

        def assign_worker_roles(workers):
            workers[0].id = 'mturk_agent_1'
            workers[1].id = 'mturk_agent_2'

        def run_task_wait():
            self.mturk_manager.start_task(
                lambda w: True, assign_worker_roles, self.get_task_world
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
        for key in self.worlds_agents.keys():
            self.worlds_agents[key] = True
        self.mturk_manager.shutdown()
        self.fake_socket.close()
        if self.task_thread.isAlive():
            self.task_thread.join()

    def get_onboard_world(self, mturk_agent):
        self.onboarding_agents[mturk_agent.worker_id] = False

        def episode_done():
            return not (
                (mturk_agent.worker_id in self.onboarding_agents)
                and (self.onboarding_agents[mturk_agent.worker_id] is False)
            )

        return TestMTurkOnboardWorld(mturk_agent, episode_done)

    def get_task_world(self, mturk_manager, opt, workers):
        for worker in workers:
            self.worlds_agents[worker.worker_id] = False

        def episode_done():
            for worker in workers:
                if self.worlds_agents[worker.worker_id] is False:
                    return False
            return True

        return TestMTurkWorld(workers, episode_done)

    def alive_agent(self, agent):
        agent.register_to_socket(self.fake_socket)
        agent.send_alive()
        time.sleep(0.3)

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

        # Assert agents move to task
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)

        # Complete agents
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        agent_1_object.set_completed_act({})
        agent_2_object.set_completed_act({})
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)

        # Assert conversation is complete for manager and agents
        assert_equal_by(lambda: manager.completed_conversations, 1, 2)

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

        # Assert agents move to task
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)

        # Disconnect agent
        agent_2.send_disconnect()
        assert_equal_by(
            agent_1_object.get_status, AssignState.STATUS_PARTNER_DISCONNECT, 3
        )
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DISCONNECT, 3)
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True

        # assert conversation not marked as complete
        self.assertEqual(manager.completed_conversations, 0)
        agent_1_object.set_completed_act({})

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
        self.onboarding_agents[agent_1.worker_id] = True

        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_EXPIRED)

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

        # Assert agents move to task
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
        assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
        self.assertIn(agent_1.worker_id, self.worlds_agents)

        # Complete agents
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        agent_1_object.set_completed_act({})
        agent_2_object.set_completed_act({})
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)

        # Assert conversation is complete for manager and agents
        assert_equal_by(lambda: manager.completed_conversations, 1, 2)

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

        # Complete agents
        self.worlds_agents[agent_1.worker_id] = True
        self.worlds_agents[agent_2.worker_id] = True
        agent_1_object.set_completed_act({})
        agent_2_object.set_completed_act({})
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
        assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)

        assert_equal_by(lambda: manager.completed_conversations, 1, 2)

    def test_no_onboard_expire_waiting(self):
        manager = self.mturk_manager
        manager.set_get_onboard_world(None)

        # Alive first agent
        agent_1 = self.agent_1
        self.alive_agent(agent_1)
        agent_1_object = manager.worker_manager.get_agent_for_assignment(
            agent_1.assignment_id
        )
        assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)

        manager._expire_agent_pool()

        self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_EXPIRED)


if __name__ == '__main__':
    unittest.main(buffer=True)
