# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""
Script for testing complete functionality of the MTurk conversation backend.
Simulates agents and interactions and tests the outcomes of interacting with
the server to ensure that the messages that are recieved are as intended.

It pretends to act in the way that core.html is supposed to follow, both
related to what is sent and recieved, what fields are checked, etc. A change
to the core.html file will not be caught by this script.

Doesn't actually interact with Amazon MTurk as they don't offer a robust
testing framework as of September 2017, so interactions with MTurk and updating
HIT status and things of the sort are not yet supported in this testing.
"""
from parlai.core.params import ParlaiParser
from parlai.mturk.core.test.integration_test.worlds import TestOnboardWorld, \
    TestSoloWorld, TestDuoWorld
from parlai.mturk.core.mturk_manager import MTurkManager, WORLD_START_TIMEOUT
from parlai.mturk.core.server_utils import setup_server, delete_server
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.worker_state import WorkerState, AssignState
from parlai.mturk.core.agents import MTURK_DISCONNECT_MESSAGE
import parlai.mturk.core.data_model as data_model
from parlai.mturk.core.mturk_utils import create_hit_config
from socketIO_client_nexus import SocketIO
import time
import os
import importlib
import copy
import uuid
import threading
from itertools import product
from joblib import Parallel, delayed

TEST_TASK_DESCRIPTION = 'This is a test task description'
MTURK_AGENT_IDS = ['TEST_USER_1', 'TEST_USER_2']
PORT = 443
FAKE_HIT_ID = 'FAKE_HIT_ID_{}'
TASK_GROUP_ID = 'TEST_TASK_GROUP_{}'
AGENT_1_ID = 'TEST_AGENT_1'
AGENT_2_ID = 'TEST_AGENT_2'
ASSIGN_1_ID = 'FAKE_ASSIGNMENT_ID_1'
HIT_1_ID = 'FAKE_HIT_ID_1'

SOCKET_TEST = 'SOCKET_TEST'
SOLO_ONBOARDING_TEST = 'SOLO_ONBOARDING_TEST'
SOLO_NO_ONBOARDING_TEST = 'SOLO_NO_ONBOARDING_TEST'
SOLO_REFRESH_TEST = 'SOLO_REFRESH_TEST'
DUO_ONBOARDING_TEST = 'DUO_ONBOARDING_TEST'
DUO_NO_ONBOARDING_TEST = 'DUO_NO_ONBOARDING_TEST'
DUO_VALID_RECONNECT_TEST = 'DUO_VALID_RECONNECT_TEST'
DUO_ONE_DISCONNECT_TEST = 'DUO_ONE_DISCONNECT_TEST'
COUNT_COMPLETE_TEST = 'COUNT_COMPLETE_TEST'
EXPIRE_HIT_TEST = 'EXPIRE_HIT_TEST'
ALLOWED_CONVERSATION_TEST = 'ALLOWED_CONVERSATION_TEST'
UNIQUE_CONVERSATION_TEST = 'UNIQUE_CONVERSATION_TEST'

FAKE_ASSIGNMENT_ID = 'FAKE_ASSIGNMENT_ID_{}_{}'
FAKE_WORKER_ID = 'FAKE_WORKER_ID_{}_{}'

DISCONNECT_WAIT_TIME = SocketManager.DEF_SOCKET_TIMEOUT + 1.5

completed_threads = {}
start_times = {}

def dummy(*args):
    pass

class MockAgent(object):
    """Class that pretends to be an MTurk agent interacting through the
    webpage by simulating the same commands that are sent from the core.html
    file. Exposes methods to use for testing and checking status
    """
    def __init__(self, opt, hit_id, assignment_id, worker_id, task_group_id):
        self.conversation_id = None
        self.id = None
        self.assignment_id = assignment_id
        self.hit_id = hit_id
        self.worker_id = worker_id
        self.some_agent_disconnected = False
        self.disconnected = False
        self.task_group_id = task_group_id
        self.socketIO = None
        self.always_beat = False
        self.ready = False
        self.wants_to_send = False

    def send_packet(self, packet):
        def callback(*args):
            pass
        event_name = data_model.SOCKET_ROUTE_PACKET_STRING
        self.socketIO.emit(event_name, packet.as_dict())

    def build_and_send_packet(self, packet_type, data, callback):
        if not callback:
            def callback(*args):
                pass

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

        self.socketIO.emit(event_name, msg, callback)

    def send_message(self, text, callback=dummy):
        if not callback:
            def callback(*args):
                pass

        data = {
            'text': text,
            'id': self.id,
            'message_id': str(uuid.uuid4()),
            'episode_done': False
        }

        self.wants_to_send = False
        self.build_and_send_packet(Packet.TYPE_MESSAGE, data, callback)

    def send_alive(self):
        data = {
            'hit_id': self.hit_id,
            'assignment_id': self.assignment_id,
            'worker_id': self.worker_id,
            'conversation_id': self.conversation_id
        }
        self.build_and_send_packet(Packet.TYPE_ALIVE, data, None)

    def setup_socket(self, server_url, message_handler):
        """Sets up a socket for an agent"""
        def on_socket_open(*args):
            self.send_alive()
        def on_new_message(*args):
            message_handler(args[0])
        def on_disconnect(*args):
            self.disconnected = True
        self.socketIO = SocketIO(server_url, PORT)
        # Register Handlers
        self.socketIO.on(data_model.SOCKET_OPEN_STRING, on_socket_open)
        self.socketIO.on(data_model.SOCKET_DISCONNECT_STRING, on_disconnect)
        self.socketIO.on(data_model.SOCKET_NEW_PACKET_STRING, on_new_message)

        # Start listening thread
        self.listen_thread = threading.Thread(target=self.socketIO.wait)
        self.listen_thread.daemon = True
        self.listen_thread.start()

    def send_heartbeat(self):
        """Sends a heartbeat to the world"""
        hb = {
            'id': str(uuid.uuid4()),
            'receiver_id': '[World_' + self.task_group_id + ']',
            'assignment_id': self.assignment_id,
            'sender_id' : self.worker_id,
            'conversation_id': self.conversation_id,
            'type': Packet.TYPE_HEARTBEAT,
            'data': None
        }
        self.socketIO.emit(data_model.SOCKET_ROUTE_PACKET_STRING, hb)

    def wait_for_alive(self):
        last_time = time.time()
        while not self.ready:
            self.send_alive()
            time.sleep(0.5)
            assert time.time() - last_time < 10, \
                'Timed out wating for server to acknowledge {} alive'.format(
                    self.worker_id
                )


def handle_setup(opt):
    """Prepare the heroku server without creating real hits"""
    create_hit_config(
        task_description=TEST_TASK_DESCRIPTION,
        unique_worker=False,
        is_sandbox=True
    )
    # Poplulate files to copy over to the server
    task_files_to_copy = []
    task_directory_path = os.path.join(
        opt['parlai_home'],
        'parlai',
        'mturk',
        'core',
        'test',
        'integration_test'
    )
    task_files_to_copy.append(
        os.path.join(task_directory_path, 'html', 'cover_page.html'))
    for mturk_agent_id in MTURK_AGENT_IDS + ['onboarding']:
        task_files_to_copy.append(os.path.join(
            task_directory_path,
            'html',
            '{}_index.html'.format(mturk_agent_id)
        ))

    # Setup the server with a likely-unique app-name
    task_name = '{}-{}'.format(str(uuid.uuid4())[:8], 'integration_test')
    server_task_name = \
        ''.join(e for e in task_name if e.isalnum() or e == '-')
    server_url = \
        setup_server(server_task_name, task_files_to_copy)

    return server_task_name, server_url


def handle_shutdown(server_task_name):
    delete_server(server_task_name)


def wait_for_state_time(seconds, mturk_manager):
    seconds_done = 0
    while (seconds_done < seconds):
        if mturk_manager.socket_manager.alive:
            seconds_done += 0.1
        time.sleep(0.1)


def run_solo_world(opt, mturk_manager, is_onboarded):
    MTURK_SOLO_WORKER = 'MTURK_SOLO_WORKER'

    # Runs the solo test world with or without onboarding
    def run_onboard(worker):
        world = TestOnboardWorld(opt=opt, mturk_agent=worker)
        while not world.episode_done():
            world.parley()
        world.shutdown()

    if is_onboarded:
        mturk_manager.set_onboard_function(onboard_function=run_onboard)
    else:
        mturk_manager.set_onboard_function(onboard_function=None)

    try:
        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            workers[0].id = MTURK_SOLO_WORKER

        global run_conversation
        def run_conversation(mturk_manager, opt, workers):
            task = opt['task']
            mturk_agent = workers[0]
            world = TestSoloWorld(opt=opt, task=task, mturk_agent=mturk_agent)
            while not world.episode_done():
                world.parley()
            world.shutdown()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )
    except:
        raise
    finally:
        pass


def run_duo_world(opt, mturk_manager, is_onboarded):
    MTURK_DUO_WORKER = 'MTURK_DUO_WORKER'

    # Runs the solo test world with or without onboarding
    def run_onboard(worker):
        world = TestOnboardWorld(opt=opt, mturk_agent=worker)
        while not world.episode_done():
            world.parley()
        world.shutdown()

    if is_onboarded:
        mturk_manager.set_onboard_function(onboard_function=run_onboard)
    else:
        mturk_manager.set_onboard_function(onboard_function=None)

    try:
        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            for worker in workers:
                worker.id = MTURK_DUO_WORKER

        global run_conversation
        def run_conversation(mturk_manager, opt, workers):
            world = TestDuoWorld(opt=opt, agents=workers)
            while not world.episode_done():
                world.parley()
            world.shutdown()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )
    except:
        raise
    finally:
        pass


def make_packet_handler_cant_task(agent, on_ack, on_hb, on_msg):
    """A packet handler that is unable to switch into task worlds"""
    def handler_mock(pkt):
        if pkt['type'] == Packet.TYPE_ACK:
            agent.ready = True
            packet = Packet.from_dict(pkt)
            on_ack(packet)
        elif pkt['type'] == Packet.TYPE_HEARTBEAT:
            packet = Packet.from_dict(pkt)
            on_hb(packet)
            time.sleep(1)
            if agent.always_beat:
                agent.send_heartbeat()
        elif pkt['type'] == Packet.TYPE_MESSAGE:
            packet = Packet.from_dict(pkt)
            if agent.always_beat:
                agent.send_packet(packet.get_ack())
            on_msg(packet)
            if packet.data['text'] == data_model.COMMAND_CHANGE_CONVERSATION:
                if not agent.always_beat:
                    pass
                elif not packet.data['conversation_id'].startswith('t_'):
                    agent.conversation_id = packet.data['conversation_id']
                    agent.id = packet.data['agent_id']
                    agent.send_alive()
                else:
                    agent.always_beat = False
        elif pkt['type'] == Packet.TYPE_ALIVE:
            raise Exception('Invalid alive packet {}'.format(pkt))
        else:
            raise Exception('Invalid Packet type {} received in {}'.format(
                pkt['type'],
                pkt
            ))
    return handler_mock


def make_packet_handler(agent, on_ack, on_hb, on_msg):
    def handler_mock(pkt):
        if pkt['type'] == Packet.TYPE_ACK:
            agent.ready = True
            packet = Packet.from_dict(pkt)
            on_ack(packet)
        elif pkt['type'] == Packet.TYPE_HEARTBEAT:
            packet = Packet.from_dict(pkt)
            on_hb(packet)
            time.sleep(1)
            if agent.always_beat:
                agent.send_heartbeat()
        elif pkt['type'] == Packet.TYPE_MESSAGE:
            packet = Packet.from_dict(pkt)
            agent.send_packet(packet.get_ack())
            on_msg(packet)
            if packet.data['text'] == data_model.COMMAND_CHANGE_CONVERSATION:
                agent.conversation_id = packet.data['conversation_id']
                agent.id = packet.data['agent_id']
                agent.send_alive()
        elif pkt['type'] == Packet.TYPE_ALIVE:
            raise Exception('Invalid alive packet {}'.format(pkt))
        else:
            raise Exception('Invalid Packet type {} received in {}'.format(
                pkt['type'],
                pkt
            ))
    return handler_mock


def check_status(input_status, desired_status):
    assert input_status == desired_status, 'Expected to be in {}, was found ' \
        'in {}'.format(desired_status, input_status)


def check_new_agent_setup(agent, mturk_manager,
                          status=AssignState.STATUS_ONBOARDING):
    mturk_agent = mturk_manager.mturk_workers[agent.worker_id]
    assert mturk_agent is not None, \
        'MTurk manager did not make a worker state on alive'
    mturk_assign = mturk_agent.agents[agent.assignment_id]
    assert mturk_assign is not None, \
        'MTurk manager did not make an assignment state on alive'
    assert mturk_assign.state.status == status, \
        'MTurk manager did not move the agent into {}, stuck in {}'.format(
            status, mturk_assign.state.status
        )
    connection_id = mturk_assign.get_connection_id()
    assert mturk_manager.socket_manager.socket_is_open(connection_id), \
        'The socket manager didn\'t open a socket for this agent'


def test_socket_manager(opt, server_url):
    global completed_threads
    TEST_MESSAGE = 'This is a test'
    task_group_id = TASK_GROUP_ID.format('TEST_SOCKET')
    socket_manager = None
    world_received_alive = False
    world_received_message = False
    agent_timed_out = False

    def world_on_alive(pkt):
        nonlocal world_received_alive
        # Assert alive packets contain the right data
        worker_id = pkt.data['worker_id']
        assert worker_id == AGENT_1_ID, 'Worker id was {}'.format(worker_id)
        hit_id = pkt.data['hit_id']
        assert hit_id == HIT_1_ID, 'HIT id was {}'.format(hit_id)
        assign_id = pkt.data['assignment_id']
        assert assign_id == ASSIGN_1_ID, 'Assign id was {}'.format(assign_id)
        conversation_id = pkt.data['conversation_id']
        assert conversation_id == None, \
            'Conversation id was {}'.format(conversation_id)
        # Start a channel
        socket_manager.open_channel(worker_id, assign_id)
        # Note that alive was successful
        world_received_alive = True

    def world_on_new_message(pkt):
        nonlocal world_received_message
        text = pkt.data['text']
        assert text == TEST_MESSAGE, 'Received text was {}'.format(text)
        world_received_message = True

    def world_on_socket_dead(worker_id, assign_id):
        nonlocal agent_timed_out
        assert worker_id == AGENT_1_ID, 'Worker id was {}'.format(worker_id)
        assert assign_id == ASSIGN_1_ID, 'Assign id was {}'.format(assign_id)
        agent_timed_out = True
        return True

    socket_manager = SocketManager(
        server_url,
        PORT,
        world_on_alive,
        world_on_new_message,
        world_on_socket_dead,
        task_group_id
    )

    agent_got_response_heartbeat = False
    received_messages = 0
    did_ack = False
    agent = MockAgent(opt, HIT_1_ID, ASSIGN_1_ID, AGENT_1_ID, task_group_id)
    connection_id = '{}_{}'.format(AGENT_1_ID, ASSIGN_1_ID)

    def agent_on_message(pkt):
        nonlocal agent_got_response_heartbeat
        nonlocal received_messages
        nonlocal agent
        if pkt['type'] == Packet.TYPE_HEARTBEAT:
            agent_got_response_heartbeat = True
        elif pkt['type'] == Packet.TYPE_MESSAGE:
            if received_messages != 0:
                packet = Packet.from_dict(pkt)
                agent.send_packet(packet.get_ack())
            received_messages += 1
        elif pkt['type'] == Packet.TYPE_ACK:
            agent.ready = True

    def manager_on_message_ack(pkt):
        nonlocal did_ack
        did_ack = True

    agent.setup_socket(server_url, agent_on_message)
    time.sleep(1)
    # Wait for socket to open to begin testing
    agent.wait_for_alive()
    assert socket_manager.socket_is_open(connection_id), \
        'Channel was not properly opened for connecting agent'

    # send some content from the agent
    time.sleep(1)
    agent.send_heartbeat()
    time.sleep(1)
    agent.send_message(TEST_MESSAGE, None)
    time.sleep(1)

    # Send some content from the socket manager, don't ack the first
    # time to ensure that resends work, and ensure the callback is
    # eventually called
    test_blocking_packet = Packet(
        'Fake_id',
        Packet.TYPE_MESSAGE,
        socket_manager.get_my_sender_id(),
        AGENT_1_ID,
        ASSIGN_1_ID,
        '',
        None,
        True,
        True,
        manager_on_message_ack
    )

    # Send packet and wait for it to arrive the first time
    socket_manager.queue_packet(test_blocking_packet)
    # Wait for socket to open to begin testing
    last_time = time.time()
    while received_messages == 0:
        time.sleep(0.5)
        assert time.time() - last_time < 10, \
            'Timed out wating for server to send message'
    assert socket_manager.get_status('Fake_id') == Packet.STATUS_SENT, \
        'Packet sent but status never updated'

    # wait for resend to occur
    time.sleep(2.5)
    assert did_ack, 'Socket_manager\'s message ack callback never fired'
    assert socket_manager.get_status('Fake_id') == Packet.STATUS_ACK, \
        'Packet recieved but status never updated'

    # Ensure queues are properly set up and that reopening an open socket
    # does nothing
    assert len(socket_manager.queues) == 1, \
        'More queues were opened than expected for the connecting agent'
    socket_manager.open_channel(AGENT_1_ID, ASSIGN_1_ID)
    assert len(socket_manager.queues) == 1, \
        'Second open for the worker was not idempotent'
    time.sleep(8.5)

    # Ensure all states happened and that the agent eventually disconnected
    assert world_received_alive, 'World never received alive message'
    assert world_received_message, 'World never received test message'
    assert agent_timed_out, 'Agent did not timeout'
    assert agent_got_response_heartbeat, 'Agent never got response heartbeat'

    # Close channels and move on
    socket_manager.close_all_channels()
    assert not socket_manager.socket_is_open(connection_id), \
        'Channel was not closed with close_all_channels'
    assert len(socket_manager.packet_map) == 0, \
        'Packets were not cleared on close, {} found'.format(
            len(socket_manager.packet_map)
        )
    assert len(socket_manager.queues) == 0, \
        'Queues were not cleared on close, {} found'.format(
            len(socket_manager.queues)
        )
    assert len(socket_manager.threads) == 0, \
        'Threads were not cleared on close, {} found'.format(
            len(socket_manager.threads)
        )

    # Test to make sure can't send a packet to a closed channel
    test_packet = Packet(
        'Fake_id',
        Packet.TYPE_MESSAGE,
        AGENT_1_ID,
        socket_manager.get_my_sender_id(),
        ASSIGN_1_ID,
        ''
    )
    socket_manager.queue_packet(test_packet)
    assert len(socket_manager.packet_map) == 0, \
        'Packets were not cleared on close, {} found'.format(
            len(socket_manager.packet_map)
        )
    completed_threads[SOCKET_TEST] = True


def test_solo_with_onboarding(opt, server_url):
    """Tests solo task with onboarding to completion, as well as disconnect in
    onboarding to ensure the agent is marked disconnected.
    """
    global completed_threads
    print('{} Starting'.format(SOLO_ONBOARDING_TEST))
    opt['task'] = SOLO_ONBOARDING_TEST
    hit_id = FAKE_HIT_ID.format(SOLO_ONBOARDING_TEST)
    assign_id_1 = FAKE_ASSIGNMENT_ID.format(SOLO_ONBOARDING_TEST, 1)
    assign_id_2 = FAKE_ASSIGNMENT_ID.format(SOLO_ONBOARDING_TEST, 2)
    worker_id = FAKE_WORKER_ID.format(SOLO_ONBOARDING_TEST, 1)
    connection_id_1 = '{}_{}'.format(worker_id, assign_id_1)
    connection_id_2 = '{}_{}'.format(worker_id, assign_id_2)
    last_command = None
    message_num = 0
    expected_messages = [
        TestOnboardWorld.TEST_TEXT_1, TestOnboardWorld.TEST_TEXT_2,
        TestSoloWorld.TEST_TEXT_1, TestSoloWorld.TEST_TEXT_2
    ]

    mturk_agent_id = AGENT_1_ID
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids = [mturk_agent_id]
    )
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_solo_world,
                                    args=(opt, mturk_manager, True))
    world_thread.daemon = True
    world_thread.start()

    # Create an agent and set it up to connect
    def msg_callback(packet):
        nonlocal last_command
        nonlocal message_num
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            last_command = packet
        else:
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    test_agent_fail = \
        MockAgent(opt, hit_id, assign_id_1, worker_id, task_group_id)
    message_handler = \
        make_packet_handler(test_agent_fail, dummy, dummy, msg_callback)
    test_agent_fail.setup_socket(server_url, message_handler)
    test_agent_fail.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_fail, mturk_manager)
    mturk_manager_assign = \
        mturk_manager.mturk_workers[worker_id].agents[assign_id_1]
    assign_state = mturk_manager_assign.state

    # Run through onboarding, then disconnect and reconnect
    test_agent_fail.always_beat = True
    test_agent_fail.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    assert test_agent_fail.conversation_id.startswith('o_'), \
        'Mock agent didn\'t make it to onboarding'
    assert last_command.data['text'] == data_model.COMMAND_SEND_MESSAGE, \
        'Agent was not asked to send message {}'.format(message_num)
    wait_for_state_time(2, mturk_manager)
    test_agent_fail.send_message('Hello1', dummy)
    test_agent_fail.always_beat = False
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)

    # Refresh the agent
    test_agent_fail.conversation_id = None
    test_agent_fail.send_alive()
    wait_for_state_time(2, mturk_manager)
    assert last_command.data['text'] == data_model.COMMAND_INACTIVE_HIT, \
        'Agent disconnected in onboarding didn\'t get inactive hit'
    assert assign_state.status == AssignState.STATUS_DISCONNECT, \
        'Disconnected agent not marked as so in state'
    assert mturk_manager_assign.disconnected == True, \
        'Disconnected agent not marked as so in agent'

    # Connect with a new agent and finish onboarding
    last_command = None
    message_num = 0
    test_agent = MockAgent(opt, hit_id, assign_id_2, worker_id, task_group_id)
    message_handler = \
        make_packet_handler(test_agent, dummy, dummy, msg_callback)
    test_agent.setup_socket(server_url, message_handler)
    test_agent.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent, mturk_manager)
    mturk_manager_assign = \
        mturk_manager.mturk_workers[worker_id].agents[assign_id_2]
    assign_state = mturk_manager_assign.state

    # Run through onboarding
    test_agent.always_beat = True
    test_agent.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    assert test_agent.conversation_id.startswith('o_'), \
        'Mock agent didn\'t make it to onboarding'
    assert last_command.data['text'] == data_model.COMMAND_SEND_MESSAGE, \
        'Agent was not asked to send message {}'.format(message_num)
    wait_for_state_time(2, mturk_manager)
    test_agent.send_message('Hello1', dummy)

    wait_for_state_time(2, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_ONBOARDING)
    test_agent.send_message('Hello2', dummy)
    wait_for_state_time(4, mturk_manager)

    # Run through task
    assert test_agent.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    last_time = time.time()
    while message_num == 2:
        # Wait for manager to catch up
        time.sleep(0.2)
        assert time.time() - last_time < 10, \
            'Timed out wating for server to acknowledge alive'

    wait_for_state_time(2, mturk_manager)
    assert last_command.data['text'] == data_model.COMMAND_SEND_MESSAGE, \
        'Agent was not asked to send message {}'.format(message_num)
    test_agent.send_message('Hello3', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_IN_TASK)
    assert mturk_manager_assign.is_in_task(), 'Manager\'s copy of agent is ' \
        'not aware that they are in a task, even though the state is'
    assert len(assign_state.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state.messages))
    test_agent.send_message('Hello4', dummy)
    test_agent.always_beat = False
    wait_for_state_time(3, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_DONE)
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_DONE)
    assert mturk_manager_assign.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_1), \
        'The socket manager didn\'t close the socket upon failure of ' \
        'onboarding, though it should have'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_2), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    assert message_num == 4, 'Not all messages were successfully processed'
    completed_threads[SOLO_ONBOARDING_TEST] = True


def test_solo_no_onboarding(opt, server_url):
    """Ensures a solo agent with no onboarding moves directly to a task world
    and is able to complete the task and be marked as completed
    """
    global completed_threads
    print('{} Starting'.format(SOLO_NO_ONBOARDING_TEST))
    opt['task'] = SOLO_NO_ONBOARDING_TEST
    hit_id = FAKE_HIT_ID.format(SOLO_NO_ONBOARDING_TEST)
    assign_id = FAKE_ASSIGNMENT_ID.format(SOLO_NO_ONBOARDING_TEST, 1)
    worker_id = FAKE_WORKER_ID.format(SOLO_NO_ONBOARDING_TEST, 1)
    last_command = None
    message_num = 0
    expected_messages = [
        TestSoloWorld.TEST_TEXT_1, TestSoloWorld.TEST_TEXT_2
    ]

    mturk_agent_id = AGENT_1_ID
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids = [mturk_agent_id]
    )
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_solo_world,
                                    args=(opt, mturk_manager, False))
    world_thread.daemon = True
    world_thread.start()

    # Create an agent and set it up to connect
    def msg_callback(packet):
        nonlocal last_command
        nonlocal message_num
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            last_command = packet
        else:
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    test_agent = MockAgent(opt, hit_id, assign_id, worker_id, task_group_id)
    message_handler = \
        make_packet_handler(test_agent, dummy, dummy, msg_callback)
    test_agent.setup_socket(server_url, message_handler)
    test_agent.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent, mturk_manager, AssignState.STATUS_IN_TASK)
    mturk_manager_assign = \
        mturk_manager.mturk_workers[worker_id].agents[assign_id]
    assign_state = mturk_manager_assign.state

    test_agent.always_beat = True
    test_agent.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Run through task
    assert test_agent.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    assert last_command.data['text'] == data_model.COMMAND_SEND_MESSAGE, \
        'Agent was not asked to send message {}'.format(message_num)
    test_agent.send_message('Hello1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_IN_TASK)
    assert len(assign_state.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state.messages))
    test_agent.send_message('Hello2', dummy)
    wait_for_state_time(3, mturk_manager)
    test_agent.always_beat = False
    check_status(assign_state.status, AssignState.STATUS_DONE)
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_DONE)
    assert len(assign_state.messages) == 0, \
        'Messages were not cleared upon completion of the task'
    assert mturk_manager_assign.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert message_num == 2, 'Not all messages were successfully processed'
    completed_threads[SOLO_NO_ONBOARDING_TEST] = True


def test_solo_refresh_in_middle(opt, server_url):
    """Tests refreshing in the middle of a solo task to make sure state is
    properly restored
    """
    global completed_threads
    print('{} Starting'.format(SOLO_REFRESH_TEST))
    opt['task'] = SOLO_REFRESH_TEST
    hit_id = FAKE_HIT_ID.format(SOLO_REFRESH_TEST)
    assign_id = FAKE_ASSIGNMENT_ID.format(SOLO_REFRESH_TEST, 1)
    worker_id = FAKE_WORKER_ID.format(SOLO_REFRESH_TEST, 1)
    last_command = None
    message_num = 0
    expected_messages = [
        TestSoloWorld.TEST_TEXT_1, TestSoloWorld.TEST_TEXT_2
    ]

    mturk_agent_id = AGENT_1_ID
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[mturk_agent_id],
        is_test=True
    )
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_solo_world,
                                    args=(opt, mturk_manager, False))
    world_thread.daemon = True
    world_thread.start()

    # Create an agent and set it up to connect
    def msg_callback(packet):
        nonlocal last_command
        nonlocal message_num
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            last_command = packet
        else:
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    test_agent = MockAgent(opt, hit_id, assign_id, worker_id, task_group_id)
    message_handler = \
        make_packet_handler(test_agent, dummy, dummy, msg_callback)
    test_agent.setup_socket(server_url, message_handler)

    test_agent.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent, mturk_manager, AssignState.STATUS_IN_TASK)
    mturk_manager_assign = \
        mturk_manager.mturk_workers[worker_id].agents[assign_id]
    assign_state = mturk_manager_assign.state

    # Run through onboarding
    test_agent.always_beat = True
    test_agent.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Run through task
    assert test_agent.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    assert last_command.data['text'] == data_model.COMMAND_SEND_MESSAGE, \
        'Agent was not asked to send message {}'.format(message_num)

    # Simulate a refresh
    test_agent.conversation_id = None
    test_agent.send_alive()

    last_time = time.time()
    while (last_command.data['text'] != data_model.COMMAND_RESTORE_STATE):
        # Wait for the restore state command
        time.sleep(1)
        assert time.time() - last_time < 10, \
            'Timed out wating for COMMAND_RESTORE_STATE to arrive'

    # Check that the restore state had what we expected
    assert test_agent.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it back to task world'
    assert len(last_command.data['messages']) == 1, \
        'State restored with more than the 1 message expected, got {}'.format(
            len(last_command.data['messages'])
        )
    assert last_command.data['messages'][0]['text'] == expected_messages[0], \
        'Message sent in restore state packet wasn\'t correct'

    test_agent.send_message('Hello1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_IN_TASK)
    assert len(assign_state.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 1'.format(len(assign_state.messages))

    test_agent.send_message('Hello2', dummy)
    test_agent.always_beat = False
    wait_for_state_time(3, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_DONE)
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_DONE)
    assert len(assign_state.messages) == 0, \
        'Messages were not cleared upon completion of the task'
    assert mturk_manager_assign.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    completed_threads[SOLO_REFRESH_TEST] = True


def test_duo_with_onboarding(opt, server_url):
    """Tests a solo task with onboarding to make sure the task doesn't begin
    until both agents are ready to go. Also tests that a third agent is not
    able to join after the conversation starts, as the HIT should be expired
    """
    global completed_threads
    print('{} Starting'.format(DUO_ONBOARDING_TEST))
    opt['task'] = DUO_ONBOARDING_TEST
    hit_id = FAKE_HIT_ID.format(DUO_ONBOARDING_TEST)
    assign_id_1 = FAKE_ASSIGNMENT_ID.format(DUO_ONBOARDING_TEST, 1)
    worker_id_1 = FAKE_WORKER_ID.format(DUO_ONBOARDING_TEST, 1)
    assign_id_2 = FAKE_ASSIGNMENT_ID.format(DUO_ONBOARDING_TEST, 2)
    # Repeat worker_id on purpose to test is_sandbox matching of unique workers
    worker_id_2 = FAKE_WORKER_ID.format(DUO_ONBOARDING_TEST, 1)
    assign_id_3 = FAKE_ASSIGNMENT_ID.format(DUO_ONBOARDING_TEST, 3)
    worker_id_3 = FAKE_WORKER_ID.format(DUO_ONBOARDING_TEST, 3)
    connection_id_1 = '{}_{}'.format(worker_id_1, assign_id_1)
    connection_id_2 = '{}_{}'.format(worker_id_2, assign_id_2)
    connection_id_3 = '{}_{}'.format(worker_id_3, assign_id_3)
    last_command = None
    message_num = 0
    expected_messages = [
        TestDuoWorld.MESSAGE_1, TestDuoWorld.MESSAGE_2,
        TestDuoWorld.MESSAGE_3, TestDuoWorld.MESSAGE_4
    ]

    mturk_agent_id_1 = AGENT_1_ID
    mturk_agent_id_2 = AGENT_2_ID
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[mturk_agent_id_1, mturk_agent_id_2],
        is_test=True
    )
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_duo_world,
                                    args=(opt, mturk_manager, True))
    world_thread.daemon = True
    world_thread.start()

    # create and set up the two agents
    test_agent_1 = MockAgent(opt, hit_id, assign_id_1,
                             worker_id_1, task_group_id)
    def msg_callback_1(packet):
        nonlocal message_num
        nonlocal test_agent_1
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_1.wants_to_send = True
        elif test_agent_1.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    test_agent_2 = MockAgent(opt, hit_id, assign_id_2,
                             worker_id_2, task_group_id)
    def msg_callback_2(packet):
        nonlocal message_num
        nonlocal test_agent_2
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_2.wants_to_send = True
        elif test_agent_2.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    test_agent_3 = MockAgent(opt, hit_id, assign_id_3,
                             worker_id_3, task_group_id)

    def msg_callback_3(packet):
        nonlocal last_command
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            last_command = packet

    message_handler_1 = \
        make_packet_handler(test_agent_1, dummy, dummy, msg_callback_1)
    message_handler_2 = \
        make_packet_handler(test_agent_2, dummy, dummy, msg_callback_2)
    message_handler_3 = \
        make_packet_handler(test_agent_3, dummy, dummy, msg_callback_3)
    test_agent_1.setup_socket(server_url, message_handler_1)
    test_agent_2.setup_socket(server_url, message_handler_2)
    test_agent_1.wait_for_alive()
    test_agent_2.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_1, mturk_manager)
    mturk_manager_assign_1 = \
        mturk_manager.mturk_workers[worker_id_1].agents[assign_id_1]
    assign_state_1 = mturk_manager_assign_1.state
    check_new_agent_setup(test_agent_2, mturk_manager)
    mturk_manager_assign_2 = \
        mturk_manager.mturk_workers[worker_id_2].agents[assign_id_2]
    assign_state_2 = mturk_manager_assign_2.state
    check_new_agent_setup(test_agent_1, mturk_manager)
    mturk_manager_assign_1 = \
        mturk_manager.mturk_workers[worker_id_1].agents[assign_id_1]
    assign_state_1 = mturk_manager_assign_1.state
    check_new_agent_setup(test_agent_2, mturk_manager)
    mturk_manager_assign_2 = \
        mturk_manager.mturk_workers[worker_id_2].agents[assign_id_2]
    assign_state_2 = mturk_manager_assign_2.state

    # Start heartbeats
    test_agent_1.always_beat = True
    test_agent_1.send_heartbeat()
    test_agent_2.always_beat = True
    test_agent_2.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Run agent_1 through onboarding
    assert test_agent_1.conversation_id.startswith('o_'), \
        'Mock agent didn\'t make it to onboarding'
    test_agent_1.send_message('Onboard1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_1.status, AssignState.STATUS_ONBOARDING)
    test_agent_1.send_message('Onboard2', dummy)
    wait_for_state_time(2, mturk_manager)

    # Ensure agent 1 is sitting in a waiting world now
    assert test_agent_1.conversation_id.startswith('w_'), \
        'Mock agent didn\'t make it to waiting'
    check_status(assign_state_1.status, AssignState.STATUS_WAITING)

    # Run agent_2 through onboarding
    assert test_agent_2.conversation_id.startswith('o_'), \
        'Mock agent didn\'t make it to onboarding'
    test_agent_2.send_message('Onboard1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_2.status, AssignState.STATUS_ONBOARDING)
    test_agent_2.send_message('Onboard2', dummy)
    wait_for_state_time(4, mturk_manager)

    # Ensure both agents are in a task world
    assert test_agent_1.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    check_status(assign_state_1.status, AssignState.STATUS_IN_TASK)
    assert test_agent_2.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    check_status(assign_state_2.status, AssignState.STATUS_IN_TASK)

    wait_for_state_time(2, mturk_manager)
    first_agent = None
    second_agent = None
    assert test_agent_1.wants_to_send or test_agent_2.wants_to_send, \
        'Neither agent is ready to send a message after arriving in task'
    if test_agent_1.wants_to_send:
        first_agent = test_agent_1
        second_agent = test_agent_2
    else:
        second_agent = test_agent_1
        first_agent = test_agent_2

    # Step through the task
    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    second_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)

    # Attempt to connect with agent 3
    assert not mturk_manager.accepting_workers, \
        'Manager shouldn\'t still be accepting workers after a conv started'
    test_agent_3.setup_socket(server_url, message_handler_3)
    test_agent_3.wait_for_alive()
    wait_for_state_time(2, mturk_manager)
    assert last_command.data['text'] == data_model.COMMAND_EXPIRE_HIT, \
        'HIT was not immediately expired when connected'

    # Finish the task
    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    assert len(assign_state_1.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_1.messages))
    assert len(assign_state_2.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_2.messages))
    second_agent.send_message(expected_messages[message_num])
    test_agent_1.always_beat = False
    test_agent_2.always_beat = False
    wait_for_state_time(3, mturk_manager)

    check_status(assign_state_1.status, AssignState.STATUS_DONE)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    check_status(assign_state_1.status, AssignState.STATUS_DONE)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)
    assert mturk_manager.completed_conversations == 1, \
        'Complete conversation not marked as complete'
    assert mturk_manager_assign_1.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert mturk_manager_assign_2.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_1), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_2), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    completed_threads[DUO_ONBOARDING_TEST] = True


def test_duo_no_onboarding(opt, server_url):
    """Tests duo task to completion, as well as disconnect in
    waiting to ensure the agent is marked disconnected and removed from pool.
    It also tests disconnect in transitioning to a world to ensure the other
    agent returns to waiting
    """
    global completed_threads
    print('{} Starting'.format(DUO_NO_ONBOARDING_TEST))
    opt['task'] = DUO_NO_ONBOARDING_TEST
    opt['count_complete'] = True
    hit_id = FAKE_HIT_ID.format(DUO_NO_ONBOARDING_TEST)
    assign_id_1 = FAKE_ASSIGNMENT_ID.format(DUO_NO_ONBOARDING_TEST, 1)
    worker_id_1 = FAKE_WORKER_ID.format(DUO_NO_ONBOARDING_TEST, 1)
    assign_id_2 = FAKE_ASSIGNMENT_ID.format(DUO_NO_ONBOARDING_TEST, 2)
    worker_id_2 = FAKE_WORKER_ID.format(DUO_NO_ONBOARDING_TEST, 2)
    assign_id_3 = FAKE_ASSIGNMENT_ID.format(DUO_NO_ONBOARDING_TEST, 3)
    worker_id_3 = FAKE_WORKER_ID.format(DUO_NO_ONBOARDING_TEST, 3)
    assign_id_4 = FAKE_ASSIGNMENT_ID.format(DUO_NO_ONBOARDING_TEST, 4)
    worker_id_4 = FAKE_WORKER_ID.format(DUO_NO_ONBOARDING_TEST, 4)
    connection_id_1 = '{}_{}'.format(worker_id_1, assign_id_1)
    connection_id_2 = '{}_{}'.format(worker_id_2, assign_id_2)
    connection_id_3 = '{}_{}'.format(worker_id_3, assign_id_3)
    connection_id_4 = '{}_{}'.format(worker_id_4, assign_id_4)
    message_num = 0
    expected_messages = [
        TestDuoWorld.MESSAGE_1, TestDuoWorld.MESSAGE_2,
        TestDuoWorld.MESSAGE_3, TestDuoWorld.MESSAGE_4
    ]

    mturk_agent_id_1 = AGENT_1_ID
    mturk_agent_id_2 = AGENT_2_ID
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[mturk_agent_id_1, mturk_agent_id_2],
        is_test=True
    )
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_duo_world,
                                    args=(opt, mturk_manager, False))
    world_thread.daemon = True
    world_thread.start()

    # create and set up an agent to disconnect when paired
    test_agent_3 = MockAgent(opt, hit_id, assign_id_3,
                             worker_id_3, task_group_id)
    def msg_callback_3(packet):
        nonlocal message_num
        nonlocal test_agent_3
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_3.wants_to_send = True

    message_handler_3 = make_packet_handler_cant_task(
        test_agent_3,
        dummy,
        dummy,
        msg_callback_3
    )
    test_agent_3.always_beat = True
    test_agent_3.setup_socket(server_url, message_handler_3)
    test_agent_3.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_3, mturk_manager,
                          AssignState.STATUS_WAITING)
    mturk_manager_assign_3 = \
        mturk_manager.mturk_workers[worker_id_3].agents[assign_id_3]
    assign_state_3 = mturk_manager_assign_3.state

    # Start heartbeats for 3
    test_agent_3.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Ensure agent 3 is sitting in a waiting world now
    assert test_agent_3.conversation_id.startswith('w_'), \
        'Mock agent didn\'t make it to waiting'
    check_status(assign_state_3.status, AssignState.STATUS_WAITING)
    assert len(mturk_manager.worker_pool) == 1, \
        'Worker was not entered into pool'

    # create and set up an agent to disconnect when returned to waiting
    test_agent_4 = MockAgent(opt, hit_id, assign_id_4,
                             worker_id_4, task_group_id)
    def msg_callback_4(packet):
        nonlocal message_num
        nonlocal test_agent_4
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_4.wants_to_send = True

    message_handler_4 = \
        make_packet_handler(test_agent_4, dummy, dummy, msg_callback_4)
    test_agent_4.setup_socket(server_url, message_handler_4)
    test_agent_4.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_4, mturk_manager,
                          AssignState.STATUS_IN_TASK)
    mturk_manager_assign_4 = \
        mturk_manager.mturk_workers[worker_id_4].agents[assign_id_4]
    assign_state_4 = mturk_manager_assign_4.state

    # Start heartbeats for 4
    test_agent_4.always_beat = True
    test_agent_4.send_heartbeat()

    assert len(mturk_manager.worker_pool) == 0, \
        'Workers were not removed from pool when assigned to a world'
    check_status(assign_state_3.status, AssignState.STATUS_ASSIGNED)

    # Wait for the world to give up on waiting
    wait_for_state_time(WORLD_START_TIMEOUT + 2.5, mturk_manager)

    # Assert that the agent is back in the waiting world
    check_status(assign_state_4.status, AssignState.STATUS_WAITING)
    assert len(mturk_manager.worker_pool) == 1, \
        'Worker was not entered returned to pool'

    # Assert that the disconnected agent is marked as so
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_3.status, AssignState.STATUS_DISCONNECT)

    # Wait for 4 to disconnect as well
    test_agent_4.always_beat = False
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    assert len(mturk_manager.worker_pool) == 0, \
        'Workers were not removed from pool when disconnected'
    check_status(assign_state_4.status, AssignState.STATUS_DISCONNECT)

    # create and set up the first successful agent
    test_agent_1 = MockAgent(opt, hit_id, assign_id_1,
                             worker_id_1, task_group_id)
    def msg_callback_1(packet):
        nonlocal message_num
        nonlocal test_agent_1
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_1.wants_to_send = True
        elif test_agent_1.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    message_handler_1 = \
        make_packet_handler(test_agent_1, dummy, dummy, msg_callback_1)
    test_agent_1.setup_socket(server_url, message_handler_1)
    test_agent_1.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_1, mturk_manager,
                          AssignState.STATUS_WAITING)
    mturk_manager_assign_1 = \
        mturk_manager.mturk_workers[worker_id_1].agents[assign_id_1]
    assign_state_1 = mturk_manager_assign_1.state

    # Start heartbeats for 1
    test_agent_1.always_beat = True
    test_agent_1.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Ensure agent 1 is sitting in a waiting world now
    assert test_agent_1.conversation_id.startswith('w_'), \
        'Mock agent didn\'t make it to waiting'
    check_status(assign_state_1.status, AssignState.STATUS_WAITING)

    # Set up the second agent
    test_agent_2 = MockAgent(opt, hit_id, assign_id_2,
                             worker_id_2, task_group_id)
    def msg_callback_2(packet):
        nonlocal message_num
        nonlocal test_agent_2
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_2.wants_to_send = True
        elif test_agent_2.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    message_handler_2 = \
        make_packet_handler(test_agent_2, dummy, dummy, msg_callback_2)
    test_agent_2.setup_socket(server_url, message_handler_2)
    test_agent_2.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_2, mturk_manager,
                          AssignState.STATUS_IN_TASK)
    mturk_manager_assign_2 = \
        mturk_manager.mturk_workers[worker_id_2].agents[assign_id_2]
    assign_state_2 = mturk_manager_assign_2.state

    # Start heartbeats for 2
    test_agent_2.always_beat = True
    test_agent_2.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Ensure both agents are in a task world
    assert test_agent_1.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    check_status(assign_state_1.status, AssignState.STATUS_IN_TASK)
    assert test_agent_2.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    check_status(assign_state_1.status, AssignState.STATUS_IN_TASK)

    first_agent = None
    second_agent = None
    assert test_agent_1.wants_to_send or test_agent_2.wants_to_send, \
        'Neither agent is ready to send a message after arriving in task'
    if test_agent_1.wants_to_send:
        first_agent = test_agent_1
        second_agent = test_agent_2
    else:
        second_agent = test_agent_1
        first_agent = test_agent_2

    # Step through the task
    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    second_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    assert len(assign_state_1.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_1.messages))
    assert len(assign_state_2.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_2.messages))
    second_agent.send_message(expected_messages[message_num])
    test_agent_1.always_beat = False
    test_agent_2.always_beat = False
    wait_for_state_time(3, mturk_manager)


    check_status(assign_state_1.status, AssignState.STATUS_DONE)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    check_status(assign_state_1.status, AssignState.STATUS_DONE)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)
    assert mturk_manager.completed_conversations == 1, \
        'Complete conversation not marked as complete'
    assert mturk_manager_assign_1.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert mturk_manager_assign_2.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_1), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_2), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    completed_threads[DUO_NO_ONBOARDING_TEST] = True


def test_duo_valid_reconnects(opt, server_url):
    """Tests reconnects during the task which should reload the conversation
    state, as well as completing a task after a reconnect.
    """
    global completed_threads
    print('{} Starting'.format(DUO_VALID_RECONNECT_TEST))
    opt['task'] = DUO_VALID_RECONNECT_TEST
    hit_id = FAKE_HIT_ID.format(DUO_VALID_RECONNECT_TEST)
    assign_id_1 = FAKE_ASSIGNMENT_ID.format(DUO_VALID_RECONNECT_TEST, 1)
    worker_id_1 = FAKE_WORKER_ID.format(DUO_VALID_RECONNECT_TEST, 1)
    assign_id_2 = FAKE_ASSIGNMENT_ID.format(DUO_VALID_RECONNECT_TEST, 2)
    worker_id_2 = FAKE_WORKER_ID.format(DUO_VALID_RECONNECT_TEST, 2)
    connection_id_1 = '{}_{}'.format(worker_id_1, assign_id_1)
    connection_id_2 = '{}_{}'.format(worker_id_2, assign_id_2)
    message_num = 0
    refresh_was_valid = False
    expected_messages = [
        TestDuoWorld.MESSAGE_1, TestDuoWorld.MESSAGE_2,
        TestDuoWorld.MESSAGE_3, TestDuoWorld.MESSAGE_4
    ]

    mturk_agent_id_1 = AGENT_1_ID
    mturk_agent_id_2 = AGENT_2_ID
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[mturk_agent_id_1, mturk_agent_id_2],
        is_test=True
    )
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_duo_world,
                                    args=(opt, mturk_manager, False))
    world_thread.daemon = True
    world_thread.start()

    # create and set up the first agent
    test_agent_1 = MockAgent(opt, hit_id, assign_id_1,
                             worker_id_1, task_group_id)
    def msg_callback_1(packet):
        nonlocal message_num
        nonlocal test_agent_1
        nonlocal refresh_was_valid
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_1.wants_to_send = True
            elif packet.data['text'] == data_model.COMMAND_RESTORE_STATE:
                messages = packet.data['messages']
                assert messages[0]['text'] == expected_messages[0], 'first ' \
                    'message in restore state {} not as expected {}'.format(
                        messages[0], expected_messages[0]
                    )
                assert messages[1]['text'] == expected_messages[1], 'second ' \
                    'message in restore state {} not as expected {}'.format(
                        messages[1], expected_messages[1]
                    )
                assert packet.data['last_command']['text'] == \
                    data_model.COMMAND_SEND_MESSAGE, 'restore state didn\'t '\
                    'include command to send a new message'
                refresh_was_valid = True
        elif test_agent_1.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    message_handler_1 = \
        make_packet_handler(test_agent_1, dummy, dummy, msg_callback_1)
    test_agent_1.setup_socket(server_url, message_handler_1)
    test_agent_1.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_1, mturk_manager,
                          AssignState.STATUS_WAITING)
    mturk_manager_assign_1 = \
        mturk_manager.mturk_workers[worker_id_1].agents[assign_id_1]
    assign_state_1 = mturk_manager_assign_1.state

    # Start heartbeats for 1
    test_agent_1.always_beat = True
    test_agent_1.send_heartbeat()
    wait_for_state_time(2, mturk_manager)

    # Ensure agent 1 is sitting in a waiting world now
    assert test_agent_1.conversation_id.startswith('w_'), \
        'Mock agent didn\'t make it to waiting'
    check_status(assign_state_1.status, AssignState.STATUS_WAITING)

    # Set up the second agent
    test_agent_2 = MockAgent(opt, hit_id, assign_id_2,
                             worker_id_2, task_group_id)
    def msg_callback_2(packet):
        nonlocal message_num
        nonlocal test_agent_2
        nonlocal refresh_was_valid
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_2.wants_to_send = True
            elif packet.data['text'] == data_model.COMMAND_RESTORE_STATE:
                messages = packet.data['messages']
                assert messages[0]['text'] == expected_messages[0], 'first ' \
                    'message in restore state {} not as expected {}'.format(
                        messages[0], expected_messages[0]
                    )
                assert messages[1]['text'] == expected_messages[1], 'second ' \
                    'message in restore state {} not as expected {}'.format(
                        messages[1], expected_messages[1]
                    )
                assert packet.data['last_command']['text'] == \
                    data_model.COMMAND_SEND_MESSAGE, 'restore state didn\'t '\
                    'include command to send a new message'
                refresh_was_valid = True
        elif test_agent_2.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    message_handler_2 = \
        make_packet_handler(test_agent_2, dummy, dummy, msg_callback_2)
    test_agent_2.setup_socket(server_url, message_handler_2)
    test_agent_2.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_2, mturk_manager,
                          AssignState.STATUS_IN_TASK)
    mturk_manager_assign_2 = \
        mturk_manager.mturk_workers[worker_id_2].agents[assign_id_2]
    assign_state_2 = mturk_manager_assign_2.state

    # Start heartbeats for 2
    test_agent_2.always_beat = True
    test_agent_2.send_heartbeat()
    wait_for_state_time(2, mturk_manager)

    # Ensure both agents are in a task world
    assert test_agent_1.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    assert test_agent_2.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    check_status(assign_state_1.status, AssignState.STATUS_IN_TASK)
    check_status(assign_state_2.status, AssignState.STATUS_IN_TASK)

    first_agent = None
    second_agent = None
    assert test_agent_1.wants_to_send or test_agent_2.wants_to_send, \
        'Neither agent is ready to send a message after arriving in task'
    if test_agent_1.wants_to_send:
        first_agent = test_agent_1
        second_agent = test_agent_2
    else:
        second_agent = test_agent_1
        first_agent = test_agent_2

    # Step through the task
    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    second_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)

    # Simulate a refresh, msg callback will verify it was valid
    first_agent.conversation_id = None
    first_agent.send_alive()
    wait_for_state_time(4, mturk_manager)
    assert refresh_was_valid, 'Information sent on refresh was invalid'

    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    assert len(assign_state_1.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_1.messages))
    assert len(assign_state_2.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_2.messages))
    second_agent.send_message(expected_messages[message_num])
    test_agent_1.always_beat = False
    test_agent_2.always_beat = False
    wait_for_state_time(3, mturk_manager)

    check_status(assign_state_1.status, AssignState.STATUS_DONE)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    check_status(assign_state_1.status, AssignState.STATUS_DONE)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)
    assert mturk_manager.completed_conversations == 1, \
        'Complete conversation not marked as complete'
    assert mturk_manager_assign_1.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert mturk_manager_assign_2.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_1), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_2), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    completed_threads[DUO_VALID_RECONNECT_TEST] = True


def test_duo_one_disconnect(opt, server_url):
    """Tests whether disconnects properly cause a task to fail and let the
    non-disconnecting partner complete the HIT. Also tests reconnecting after
    a partner disconnect or after a disconnect.
    """
    global completed_threads
    print('{} Starting'.format(DUO_ONE_DISCONNECT_TEST))
    opt['task'] = DUO_ONE_DISCONNECT_TEST
    hit_id = FAKE_HIT_ID.format(DUO_ONE_DISCONNECT_TEST)
    assign_id_1 = FAKE_ASSIGNMENT_ID.format(DUO_ONE_DISCONNECT_TEST, 1)
    worker_id_1 = FAKE_WORKER_ID.format(DUO_ONE_DISCONNECT_TEST, 1)
    assign_id_2 = FAKE_ASSIGNMENT_ID.format(DUO_ONE_DISCONNECT_TEST, 2)
    worker_id_2 = FAKE_WORKER_ID.format(DUO_ONE_DISCONNECT_TEST, 2)
    connection_id_1 = '{}_{}'.format(worker_id_1, assign_id_1)
    connection_id_2 = '{}_{}'.format(worker_id_2, assign_id_2)
    message_num = 0
    partner_disconnects = 0
    self_disconnects = 0
    expected_messages = [
        TestDuoWorld.MESSAGE_1, TestDuoWorld.MESSAGE_2,
        MTURK_DISCONNECT_MESSAGE
    ]

    mturk_agent_id_1 = AGENT_1_ID
    mturk_agent_id_2 = AGENT_2_ID
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[mturk_agent_id_1, mturk_agent_id_2],
        is_test=True
    )
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_duo_world,
                                    args=(opt, mturk_manager, False))
    world_thread.daemon = True
    world_thread.start()

    # create and set up the first agent
    test_agent_1 = MockAgent(opt, hit_id, assign_id_1,
                             worker_id_1, task_group_id)
    def msg_callback_1(packet):
        nonlocal message_num
        nonlocal test_agent_1
        nonlocal partner_disconnects
        nonlocal self_disconnects
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_1.wants_to_send = True
            elif packet.data['text'] == data_model.COMMAND_INACTIVE_DONE:
                partner_disconnects += 1
            elif packet.data['text'] == data_model.COMMAND_INACTIVE_HIT:
                self_disconnects += 1
        elif test_agent_1.conversation_id is not None and \
                test_agent_1.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    message_handler_1 = \
        make_packet_handler(test_agent_1, dummy, dummy, msg_callback_1)
    test_agent_1.setup_socket(server_url, message_handler_1)
    test_agent_1.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_1, mturk_manager,
                          AssignState.STATUS_WAITING)
    mturk_manager_assign_1 = \
        mturk_manager.mturk_workers[worker_id_1].agents[assign_id_1]
    assign_state_1 = mturk_manager_assign_1.state

    # Start heartbeats for 1
    test_agent_1.always_beat = True
    test_agent_1.send_heartbeat()
    wait_for_state_time(2, mturk_manager)

    # Ensure agent 1 is sitting in a waiting world now
    assert test_agent_1.conversation_id.startswith('w_'), \
        'Mock agent didn\'t make it to waiting'
    check_status(assign_state_1.status, AssignState.STATUS_WAITING)

    # Set up the second agent
    test_agent_2 = MockAgent(opt, hit_id, assign_id_2,
                             worker_id_2, task_group_id)
    def msg_callback_2(packet):
        nonlocal message_num
        nonlocal test_agent_2
        nonlocal partner_disconnects
        nonlocal self_disconnects
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_2.wants_to_send = True
            elif packet.data['text'] == data_model.COMMAND_INACTIVE_DONE:
                partner_disconnects += 1
            elif packet.data['text'] == data_model.COMMAND_INACTIVE_HIT:
                self_disconnects += 1
        elif test_agent_2.conversation_id is not None and \
                test_agent_2.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    message_handler_2 = \
        make_packet_handler(test_agent_2, dummy, dummy, msg_callback_2)
    test_agent_2.setup_socket(server_url, message_handler_2)
    test_agent_2.wait_for_alive()
    wait_for_state_time(2.5, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_2, mturk_manager,
                          AssignState.STATUS_IN_TASK)
    mturk_manager_assign_2 = \
        mturk_manager.mturk_workers[worker_id_2].agents[assign_id_2]
    assign_state_2 = mturk_manager_assign_2.state

    # Start heartbeats for 2
    test_agent_2.always_beat = True
    test_agent_2.send_heartbeat()
    wait_for_state_time(2.5, mturk_manager)

    # Ensure both agents are in a task world
    assert test_agent_1.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    assert test_agent_2.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    check_status(assign_state_1.status, AssignState.STATUS_IN_TASK)
    check_status(assign_state_2.status, AssignState.STATUS_IN_TASK)

    first_agent = None
    second_agent = None
    mturk_first_agent = None
    mturk_second_agent = None
    assert test_agent_1.wants_to_send or test_agent_2.wants_to_send, \
        'Neither agent is ready to send a message after arriving in task'
    if test_agent_1.wants_to_send:
        first_agent = test_agent_1
        second_agent = test_agent_2
        mturk_first_agent = mturk_manager_assign_1
        mturk_second_agent = mturk_manager_assign_2
    else:
        second_agent = test_agent_1
        first_agent = test_agent_2
        mturk_second_agent = mturk_manager_assign_1
        mturk_first_agent = mturk_manager_assign_2

    # Step through the task
    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    second_agent.send_message(expected_messages[message_num])
    # Disconnect the first agent
    first_agent.always_beat = False
    wait_for_state_time(2, mturk_manager)

    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)

    assert partner_disconnects == 1, \
        'Connected agent did not recieve an inactive_done command'

    # Refresh the second agent
    second_agent.conversation_id = None
    second_agent.send_alive()
    wait_for_state_time(2, mturk_manager)
    assert partner_disconnects == 2, \
        'Reconnected agent did not recieve an inactive_done command'

    # Refresh the first agent
    first_agent.conversation_id = None
    first_agent.send_alive()
    wait_for_state_time(2, mturk_manager)
    assert self_disconnects == 1, \
        'Disconnected agent did not recieve an inactive command'

    # Disconnect the second agent
    second_agent.always_beat = False
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)


    check_status(mturk_second_agent.state.status,
        AssignState.STATUS_PARTNER_DISCONNECT)
    check_status(mturk_first_agent.state.status,
        AssignState.STATUS_DISCONNECT)
    assert mturk_manager.completed_conversations == 0, \
        'Incomplete conversation marked as complete'
    assert mturk_second_agent.disconnected == False, \
        'MTurk manager improperly marked the connected agent as disconnected'
    assert mturk_first_agent.disconnected == True, \
        'MTurk did not mark the disconnected agent as so'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_1), \
        'The socket manager didn\'t close the socket upon failure of the ' \
        'task, though it should have'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_2), \
        'The socket manager didn\'t close the socket upon failure of the ' \
        'task, though it should have'
    completed_threads[DUO_ONE_DISCONNECT_TEST] = True


def test_count_complete(opt, server_url):
    """Starts two worlds even though only one is requested by using the
    count_complete flag.
    """
    global completed_threads
    print('{} Starting'.format(COUNT_COMPLETE_TEST))
    opt['task'] = COUNT_COMPLETE_TEST
    opt['count_complete'] = True
    opt['num_conversations'] = 1
    hit_id = FAKE_HIT_ID.format(COUNT_COMPLETE_TEST)
    assign_id_1 = FAKE_ASSIGNMENT_ID.format(COUNT_COMPLETE_TEST, 1)
    worker_id_1 = FAKE_WORKER_ID.format(COUNT_COMPLETE_TEST, 1)
    assign_id_2 = FAKE_ASSIGNMENT_ID.format(COUNT_COMPLETE_TEST, 2)
    worker_id_2 = FAKE_WORKER_ID.format(COUNT_COMPLETE_TEST, 2)
    last_command = None
    message_num_1 = 0
    message_num_2 = 0
    expected_messages = [TestSoloWorld.TEST_TEXT_1, TestSoloWorld.TEST_TEXT_2]

    mturk_agent_id = AGENT_1_ID
    mturk_manager = MTurkManager(opt=opt,
                                 mturk_agent_ids=[mturk_agent_id],
                                 is_test=True)
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_solo_world,
                                    args=(opt, mturk_manager, False))
    world_thread.daemon = True
    world_thread.start()

    # Create an agent and set it up to connect
    def msg_callback_1(packet):
        nonlocal last_command
        nonlocal message_num_1
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            last_command = packet
        else:
            assert packet.data['text'] == expected_messages[message_num_1], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num_1],
                    message_num_1,
                    packet.data['text']
                )
            message_num_1 += 1

    test_agent_1 = \
        MockAgent(opt, hit_id, assign_id_1, worker_id_1, task_group_id)
    message_handler = \
        make_packet_handler(test_agent_1, dummy, dummy, msg_callback_1)
    test_agent_1.setup_socket(server_url, message_handler)
    test_agent_1.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_1, mturk_manager,
                          AssignState.STATUS_IN_TASK)
    mturk_manager_assign_1 = \
        mturk_manager.mturk_workers[worker_id_1].agents[assign_id_1]
    assign_state_1 = mturk_manager_assign_1.state

    # Run through onboarding
    test_agent_1.always_beat = True
    test_agent_1.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Run through task
    assert test_agent_1.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    assert last_command.data['text'] == data_model.COMMAND_SEND_MESSAGE, \
        'Agent was not asked to send message {}'.format(message_num)
    test_agent_1.send_message('Hello1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_1.status, AssignState.STATUS_IN_TASK)
    assert len(assign_state_1.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_1.messages))

    # Start the second agent while the first is still waiting
    def msg_callback_2(packet):
        nonlocal last_command
        nonlocal message_num_2
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            last_command = packet
        else:
            assert packet.data['text'] == expected_messages[message_num_2], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num_2],
                    message_num_2,
                    packet.data['text']
                )
            message_num_2 += 1

    test_agent_2 = \
        MockAgent(opt, hit_id, assign_id_2, worker_id_2, task_group_id)
    message_handler = \
        make_packet_handler(test_agent_2, dummy, dummy, msg_callback_2)
    test_agent_2.setup_socket(server_url, message_handler)
    test_agent_2.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_2, mturk_manager,
                          AssignState.STATUS_IN_TASK)
    mturk_manager_assign_2 = \
        mturk_manager.mturk_workers[worker_id_2].agents[assign_id_2]
    assign_state_2 = mturk_manager_assign_2.state

    # Run through onboarding
    test_agent_2.always_beat = True
    test_agent_2.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Run through task
    assert test_agent_2.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    assert last_command.data['text'] == data_model.COMMAND_SEND_MESSAGE, \
        'Agent was not asked to send message {}'.format(message_num)
    test_agent_2.send_message('Hello1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_2.status, AssignState.STATUS_IN_TASK)
    assert len(assign_state_2.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_2.messages))
    test_agent_2.send_message('Hello2', dummy)
    test_agent_2.always_beat = False

    # Finish agent 1's task
    test_agent_1.send_message('Hello2', dummy)
    test_agent_1.always_beat = False
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_1.status, AssignState.STATUS_DONE)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)

    # Wait for both to disconnect
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    check_status(assign_state_1.status, AssignState.STATUS_DONE)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)
    assert len(assign_state_1.messages) == 0, \
        'Messages were not cleared upon completion of the task'
    assert len(assign_state_2.messages) == 0, \
        'Messages were not cleared upon completion of the task'
    assert mturk_manager_assign_1.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert mturk_manager_assign_2.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert mturk_manager.started_conversations == 2, \
        'At least one conversation wasn\'t successfully logged'
    assert mturk_manager.completed_conversations == 2, \
        'At least one conversation wasn\'t successfully logged'
    assert message_num_1 == 2, 'Not all messages were successfully processed'
    assert message_num_2 == 2, 'Not all messages were successfully processed'
    completed_threads[COUNT_COMPLETE_TEST] = True
    pass


def test_expire_hit(opt, server_url):
    """Tests force_expire_hit by creating 4 workers, leaving
    one in onboarding and sending 3 to waiting, then ensuring that the
    remaining waiting worker gets expired"""
    global completed_threads
    print('{} Starting'.format(EXPIRE_HIT_TEST))
    opt['task'] = EXPIRE_HIT_TEST
    opt['count_complete'] = True
    hit_id = FAKE_HIT_ID.format(EXPIRE_HIT_TEST)
    assign_id_1 = FAKE_ASSIGNMENT_ID.format(EXPIRE_HIT_TEST, 1)
    worker_id_1 = FAKE_WORKER_ID.format(EXPIRE_HIT_TEST, 1)
    assign_id_2 = FAKE_ASSIGNMENT_ID.format(EXPIRE_HIT_TEST, 2)
    worker_id_2 = FAKE_WORKER_ID.format(EXPIRE_HIT_TEST, 2)
    assign_id_3 = FAKE_ASSIGNMENT_ID.format(EXPIRE_HIT_TEST, 3)
    worker_id_3 = FAKE_WORKER_ID.format(EXPIRE_HIT_TEST, 3)
    assign_id_4 = FAKE_ASSIGNMENT_ID.format(EXPIRE_HIT_TEST, 4)
    worker_id_4 = FAKE_WORKER_ID.format(EXPIRE_HIT_TEST, 4)
    connection_id_1 = '{}_{}'.format(worker_id_1, assign_id_1)
    connection_id_2 = '{}_{}'.format(worker_id_2, assign_id_2)
    connection_id_3 = '{}_{}'.format(worker_id_3, assign_id_3)
    connection_id_4 = '{}_{}'.format(worker_id_4, assign_id_4)
    last_command_3 = None
    last_command_4 = None
    message_num = 0
    expected_messages = [
        TestDuoWorld.MESSAGE_1, TestDuoWorld.MESSAGE_2,
        TestDuoWorld.MESSAGE_3, TestDuoWorld.MESSAGE_4
    ]

    mturk_agent_id_1 = AGENT_1_ID
    mturk_agent_id_2 = AGENT_2_ID
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[mturk_agent_id_1, mturk_agent_id_2],
        is_test=True
    )
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_duo_world,
                                    args=(opt, mturk_manager, True))
    world_thread.daemon = True
    world_thread.start()

    # create and set up the two agents
    test_agent_1 = MockAgent(opt, hit_id, assign_id_1,
                             worker_id_1, task_group_id)
    def msg_callback_1(packet):
        nonlocal message_num
        nonlocal test_agent_1
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_1.wants_to_send = True
        elif test_agent_1.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    test_agent_2 = MockAgent(opt, hit_id, assign_id_2,
                             worker_id_2, task_group_id)
    def msg_callback_2(packet):
        nonlocal message_num
        nonlocal test_agent_2
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_2.wants_to_send = True
        elif test_agent_2.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    test_agent_3 = MockAgent(opt, hit_id, assign_id_3,
                             worker_id_3, task_group_id)

    def msg_callback_3(packet):
        nonlocal last_command_3
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            last_command_3 = packet


    test_agent_4 = MockAgent(opt, hit_id, assign_id_4,
                             worker_id_4, task_group_id)

    def msg_callback_4(packet):
        nonlocal last_command_4
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            last_command_4 = packet

    message_handler_1 = \
        make_packet_handler(test_agent_1, dummy, dummy, msg_callback_1)
    message_handler_2 = \
        make_packet_handler(test_agent_2, dummy, dummy, msg_callback_2)
    message_handler_3 = \
        make_packet_handler(test_agent_3, dummy, dummy, msg_callback_3)
    message_handler_4 = \
        make_packet_handler(test_agent_4, dummy, dummy, msg_callback_4)
    test_agent_1.setup_socket(server_url, message_handler_1)
    test_agent_2.setup_socket(server_url, message_handler_2)
    test_agent_3.setup_socket(server_url, message_handler_3)
    test_agent_4.setup_socket(server_url, message_handler_4)
    test_agent_1.wait_for_alive()
    test_agent_2.wait_for_alive()
    test_agent_3.wait_for_alive()
    test_agent_4.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_1, mturk_manager)
    mturk_manager_assign_1 = \
        mturk_manager.mturk_workers[worker_id_1].agents[assign_id_1]
    assign_state_1 = mturk_manager_assign_1.state
    check_new_agent_setup(test_agent_2, mturk_manager)
    mturk_manager_assign_2 = \
        mturk_manager.mturk_workers[worker_id_2].agents[assign_id_2]
    assign_state_2 = mturk_manager_assign_2.state
    check_new_agent_setup(test_agent_3, mturk_manager)
    mturk_manager_assign_3 = \
        mturk_manager.mturk_workers[worker_id_3].agents[assign_id_3]
    assign_state_3 = mturk_manager_assign_3.state
    check_new_agent_setup(test_agent_4, mturk_manager)
    mturk_manager_assign_4 = \
        mturk_manager.mturk_workers[worker_id_4].agents[assign_id_4]
    assign_state_4 = mturk_manager_assign_4.state

    # Start heartbeats
    test_agent_1.always_beat = True
    test_agent_1.send_heartbeat()
    test_agent_2.always_beat = True
    test_agent_2.send_heartbeat()
    test_agent_3.always_beat = True
    test_agent_3.send_heartbeat()
    test_agent_4.always_beat = True
    test_agent_4.send_heartbeat()
    wait_for_state_time(2, mturk_manager)

    # Run agent_1 through onboarding
    assert test_agent_1.conversation_id.startswith('o_'), \
        'Mock agent didn\'t make it to onboarding'
    test_agent_1.send_message('Onboard1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_1.status, AssignState.STATUS_ONBOARDING)
    test_agent_1.send_message('Onboard2', dummy)
    wait_for_state_time(3, mturk_manager)

    # Ensure agent 1 is sitting in a waiting world now
    assert test_agent_1.conversation_id.startswith('w_'), \
        'Mock agent didn\'t make it to waiting'
    check_status(assign_state_1.status, AssignState.STATUS_WAITING)

    # Run agent_2 through onboarding
    assert test_agent_2.conversation_id.startswith('o_'), \
        'Mock agent didn\'t make it to onboarding'
    test_agent_2.send_message('Onboard1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_2.status, AssignState.STATUS_ONBOARDING)
    test_agent_2.send_message('Onboard2', dummy)
    wait_for_state_time(3, mturk_manager)

    # Ensure both agents are in a task world
    assert test_agent_1.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to waiting'
    check_status(assign_state_1.status, AssignState.STATUS_IN_TASK)
    assert test_agent_2.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to waiting'
    check_status(assign_state_2.status, AssignState.STATUS_IN_TASK)

    # Run agent_3 through onboarding
    assert test_agent_3.conversation_id.startswith('o_'), \
        'Mock agent didn\'t make it to onboarding'
    test_agent_3.send_message('Onboard1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_3.status, AssignState.STATUS_ONBOARDING)
    test_agent_3.send_message('Onboard2', dummy)
    wait_for_state_time(2, mturk_manager)

    # Ensure agent 3 is sitting in a waiting world now
    assert test_agent_3.conversation_id.startswith('w_'), \
        'Mock agent didn\'t make it to waiting'
    check_status(assign_state_3.status, AssignState.STATUS_WAITING)

    wait_for_state_time(2, mturk_manager)
    first_agent = None
    second_agent = None
    assert test_agent_1.wants_to_send or test_agent_2.wants_to_send, \
        'Neither agent is ready to send a message after arriving in task'
    if test_agent_1.wants_to_send:
        first_agent = test_agent_1
        second_agent = test_agent_2
    else:
        second_agent = test_agent_1
        first_agent = test_agent_2

    # Step through the task
    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    second_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    assert len(assign_state_1.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_1.messages))
    assert len(assign_state_2.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_2.messages))
    second_agent.send_message(expected_messages[message_num])
    test_agent_1.always_beat = False
    test_agent_2.always_beat = False
    wait_for_state_time(5, mturk_manager)

    # Assert that the two other agents were expired
    check_status(assign_state_3.status, AssignState.STATUS_EXPIRED)
    check_status(assign_state_4.status, AssignState.STATUS_EXPIRED)
    assert last_command_3.data['text'] == data_model.COMMAND_EXPIRE_HIT, \
        'Waiting world agent was not expired'
    assert last_command_4.data['text'] == data_model.COMMAND_EXPIRE_HIT, \
        'Onboarding world agent was not expired'
    test_agent_3.always_beat = False
    test_agent_4.always_beat = False


    check_status(assign_state_1.status, AssignState.STATUS_DONE)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    check_status(assign_state_1.status, AssignState.STATUS_DONE)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)
    assert mturk_manager.completed_conversations == 1, \
        'Complete conversation not marked as complete'
    assert mturk_manager_assign_1.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert mturk_manager_assign_2.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert mturk_manager_assign_3.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert mturk_manager_assign_4.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert mturk_manager_assign_3.hit_is_expired == True, \
        'MTurk manager failed to mark agent as expired'
    assert mturk_manager_assign_4.hit_is_expired == True, \
        'MTurk manager failed to mark agent as expired'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_1), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_2), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_3), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_4), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    completed_threads[EXPIRE_HIT_TEST] = True


def test_allowed_conversations(opt, server_url):
    """Test to ensure that an agent can't take part in two conversations at
    the same time when only one concurrent conversation is allowed, but that
    they're allowed to start it after finishing the first
    """
    global completed_threads
    print('{} Starting'.format(ALLOWED_CONVERSATION_TEST))
    opt['allowed_conversations'] = 1
    opt['num_conversations'] = 2
    opt['task'] = ALLOWED_CONVERSATION_TEST
    hit_id = FAKE_HIT_ID.format(ALLOWED_CONVERSATION_TEST)
    assign_id = FAKE_ASSIGNMENT_ID.format(ALLOWED_CONVERSATION_TEST, 1)
    assign_id_2 = FAKE_ASSIGNMENT_ID.format(ALLOWED_CONVERSATION_TEST, 2)
    worker_id = FAKE_WORKER_ID.format(ALLOWED_CONVERSATION_TEST, 1)
    last_command = None
    message_num = 0
    expected_messages = [
        TestSoloWorld.TEST_TEXT_1, TestSoloWorld.TEST_TEXT_2
    ]

    mturk_agent_id = AGENT_1_ID
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[mturk_agent_id],
        is_test=True
    )
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_solo_world,
                                    args=(opt, mturk_manager, False))
    world_thread.daemon = True
    world_thread.start()

    # Create an agent and set it up to connect
    def msg_callback(packet):
        nonlocal last_command
        nonlocal message_num
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            last_command = packet
        else:
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    test_agent = MockAgent(opt, hit_id, assign_id, worker_id, task_group_id)
    message_handler = \
        make_packet_handler(test_agent, dummy, dummy, msg_callback)
    test_agent.setup_socket(server_url, message_handler)
    test_agent.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent, mturk_manager, AssignState.STATUS_IN_TASK)
    mturk_manager_assign = \
        mturk_manager.mturk_workers[worker_id].agents[assign_id]
    assign_state = mturk_manager_assign.state
    test_agent.always_beat = True
    test_agent.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Run through task
    assert test_agent.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    assert last_command.data['text'] == data_model.COMMAND_SEND_MESSAGE, \
        'Agent was not asked to send message {}'.format(message_num)
    test_agent.send_message('Hello1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_IN_TASK)
    assert len(assign_state.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state.messages))

    # Try to connect to second conversation
    test_agent_2 = \
        MockAgent(opt, hit_id, assign_id_2, worker_id, task_group_id)
    message_handler = \
        make_packet_handler(test_agent_2, dummy, dummy, msg_callback)
    test_agent_2.setup_socket(server_url, message_handler)
    test_agent_2.wait_for_alive()
    wait_for_state_time(2, mturk_manager)
    assert last_command.data['text'] == data_model.COMMAND_EXPIRE_HIT, \
        'HIT was not immediately expired when connected'

    # Finish first conversation
    test_agent.send_message('Hello2', dummy)
    test_agent.always_beat = False
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_DONE)

    # Retry second conversation
    last_command = None
    message_num = 0
    test_agent_2.send_alive()
    test_agent_2.always_beat = False
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_2, mturk_manager, AssignState.STATUS_IN_TASK)
    mturk_manager_assign_2 = \
        mturk_manager.mturk_workers[worker_id].agents[assign_id_2]
    assign_state_2 = mturk_manager_assign_2.state
    test_agent_2.always_beat = True
    test_agent_2.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Run through task
    assert test_agent_2.conversation_id.startswith('t_'), \
        'Mock agent didn\'t make it to task world'
    assert last_command.data['text'] == data_model.COMMAND_SEND_MESSAGE, \
        'Agent was not asked to send message {}'.format(message_num)
    test_agent_2.send_message('Hello1', dummy)
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_2.status, AssignState.STATUS_IN_TASK)
    assert len(assign_state_2.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_2.messages))
    test_agent_2.send_message('Hello2', dummy)
    test_agent_2.always_beat = False
    wait_for_state_time(2, mturk_manager)
    check_status(assign_state_2.status, AssignState.STATUS_DONE)


    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    check_status(assign_state.status, AssignState.STATUS_DONE)
    assert len(assign_state.messages) == 0, \
        'Messages were not cleared upon completion of the task'
    assert mturk_manager_assign.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert message_num == 2, 'Not all messages were successfully processed'
    completed_threads[ALLOWED_CONVERSATION_TEST] = True


def test_unique_workers_in_conversation(opt, server_url):
    """Ensures that a worker cannot start a conversation with themselves
    when not in the sandbox
    """
    global completed_threads
    print('{} Starting'.format(UNIQUE_CONVERSATION_TEST))
    opt['task'] = UNIQUE_CONVERSATION_TEST
    opt['is_sandbox'] = False
    opt['count_complete'] = True
    hit_id = FAKE_HIT_ID.format(UNIQUE_CONVERSATION_TEST)
    worker_id_1 = FAKE_WORKER_ID.format(UNIQUE_CONVERSATION_TEST, 1)
    worker_id_2 = FAKE_WORKER_ID.format(UNIQUE_CONVERSATION_TEST, 2)
    assign_id_1 = FAKE_ASSIGNMENT_ID.format(UNIQUE_CONVERSATION_TEST, 1)
    assign_id_2 = FAKE_ASSIGNMENT_ID.format(UNIQUE_CONVERSATION_TEST, 2)
    assign_id_3 = FAKE_ASSIGNMENT_ID.format(UNIQUE_CONVERSATION_TEST, 3)
    connection_id_1 = '{}_{}'.format(worker_id_1, assign_id_1)
    connection_id_2 = '{}_{}'.format(worker_id_1, assign_id_2)
    connection_id_3 = '{}_{}'.format(worker_id_2, assign_id_3)
    message_num = 0
    expected_messages = [
        TestDuoWorld.MESSAGE_1, TestDuoWorld.MESSAGE_2,
        TestDuoWorld.MESSAGE_3, TestDuoWorld.MESSAGE_4
    ]

    mturk_agent_id_1 = AGENT_1_ID
    mturk_agent_id_2 = AGENT_2_ID
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=[mturk_agent_id_1, mturk_agent_id_2],
        is_test=True
    )
    mturk_manager.server_url = server_url
    mturk_manager.start_new_run()
    task_group_id = mturk_manager.task_group_id
    world_thread = threading.Thread(target=run_duo_world,
                                    args=(opt, mturk_manager, False))
    world_thread.daemon = True
    world_thread.start()

    # create and set up the two agents for the one worker
    test_agent_1 = MockAgent(opt, hit_id, assign_id_1,
                             worker_id_1, task_group_id)
    def msg_callback_1(packet):
        nonlocal message_num
        nonlocal test_agent_1
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_1.wants_to_send = True
        elif test_agent_1.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    message_handler_1 = \
        make_packet_handler(test_agent_1, dummy, dummy, msg_callback_1)
    test_agent_1.setup_socket(server_url, message_handler_1)
    test_agent_1.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_1, mturk_manager,
                          AssignState.STATUS_WAITING)
    mturk_manager_assign_1 = \
        mturk_manager.mturk_workers[worker_id_1].agents[assign_id_1]
    assign_state_1 = mturk_manager_assign_1.state

    # Start heartbeats for 1
    test_agent_1.always_beat = True
    test_agent_1.send_heartbeat()
    wait_for_state_time(3, mturk_manager)

    # Ensure agent 1 is sitting in a waiting world now
    assert test_agent_1.conversation_id.startswith('w_'), \
        'Mock agent didn\'t make it to waiting'
    check_status(assign_state_1.status, AssignState.STATUS_WAITING)

    # Set up the second agent
    test_agent_2 = MockAgent(opt, hit_id, assign_id_2,
                             worker_id_1, task_group_id)
    def msg_callback_2(packet):
        nonlocal message_num
        nonlocal test_agent_2
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_2.wants_to_send = True
        elif test_agent_2.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    message_handler_2 = \
        make_packet_handler(test_agent_2, dummy, dummy, msg_callback_2)
    test_agent_2.setup_socket(server_url, message_handler_2)
    test_agent_2.wait_for_alive()
    wait_for_state_time(3, mturk_manager)

    # Ensure no task has started yet
    assert test_agent_2.conversation_id.startswith('w_'), \
        'Mock agent didn\'t make it to waiting'
    mturk_manager_assign_2 = \
        mturk_manager.mturk_workers[worker_id_1].agents[assign_id_2]
    assign_state_2 = mturk_manager_assign_2.state
    check_status(assign_state_1.status, AssignState.STATUS_WAITING)
    check_status(assign_state_2.status, AssignState.STATUS_WAITING)

    # Start heartbeats for 2
    test_agent_2.always_beat = True
    test_agent_2.send_heartbeat()
    wait_for_state_time(2, mturk_manager)

    # Create third agent
    test_agent_3 = MockAgent(opt, hit_id, assign_id_3,
                             worker_id_2, task_group_id)
    def msg_callback_3(packet):
        nonlocal message_num
        nonlocal test_agent_3
        if packet.data['type'] == data_model.MESSAGE_TYPE_COMMAND:
            if packet.data['text'] == data_model.COMMAND_SEND_MESSAGE:
                test_agent_3.wants_to_send = True
        elif test_agent_1.conversation_id.startswith('t_'):
            assert packet.data['text'] == expected_messages[message_num], \
                'Expected {} for message {}, got {}'.format(
                    expected_messages[message_num],
                    message_num,
                    packet.data['text']
                )
            message_num += 1

    message_handler_3 = \
        make_packet_handler(test_agent_3, dummy, dummy, msg_callback_3)
    test_agent_3.setup_socket(server_url, message_handler_3)
    test_agent_3.wait_for_alive()
    wait_for_state_time(2, mturk_manager)

    # Start heartbeats for 3
    test_agent_3.always_beat = True
    test_agent_3.send_heartbeat()

    # Assert that the state was properly set up
    check_new_agent_setup(test_agent_3, mturk_manager,
                          AssignState.STATUS_IN_TASK)
    mturk_manager_assign_3 = \
        mturk_manager.mturk_workers[worker_id_2].agents[assign_id_3]
    assign_state_3 = mturk_manager_assign_3.state

    in_agent = None
    in_assign = None
    out_agent = None
    out_assign = None
    if assign_state_1.status == AssignState.STATUS_IN_TASK:
        in_agent = test_agent_1
        in_assign = mturk_manager_assign_1
        out_agent = test_agent_2
        out_assign = mturk_manager_assign_2
    elif assign_state_2.status == AssignState.STATUS_IN_TASK:
        out_agent = test_agent_1
        out_assign = mturk_manager_assign_1
        in_agent = test_agent_2
        in_assign = mturk_manager_assign_2
    else:
        assert False, 'Neither agent moved into the task world'

    wait_for_state_time(4, mturk_manager)
    assert in_agent.wants_to_send or test_agent_3.wants_to_send, \
        'Neither agent is ready to send a message after arriving in task'

    first_agent = None
    second_agent = None
    if in_agent.wants_to_send:
        first_agent = in_agent
        second_agent = test_agent_3
    else:
        first_agent = test_agent_3
        second_agent = in_agent

    # Step through the task
    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    second_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    first_agent.send_message(expected_messages[message_num])
    wait_for_state_time(2, mturk_manager)
    assert len(in_assign.state.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_1.messages))
    assert len(assign_state_3.messages) == 3, \
        'Not all of the messages have been stored into the state, found {}' \
        'when expecting 3'.format(len(assign_state_2.messages))
    second_agent.send_message(expected_messages[message_num])
    test_agent_1.always_beat = False
    test_agent_2.always_beat = False
    wait_for_state_time(3, mturk_manager)

    check_status(in_assign.state.status, AssignState.STATUS_DONE)
    check_status(out_assign.state.status, AssignState.STATUS_EXPIRED)
    check_status(assign_state_3.status, AssignState.STATUS_DONE)
    wait_for_state_time(DISCONNECT_WAIT_TIME, mturk_manager)
    check_status(in_assign.state.status, AssignState.STATUS_DONE)
    check_status(out_assign.state.status, AssignState.STATUS_EXPIRED)
    check_status(assign_state_3.status, AssignState.STATUS_DONE)
    assert mturk_manager.completed_conversations == 1, \
        'Complete conversation not marked as complete'
    assert in_assign.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert out_assign.disconnected == False, \
        'MTurk manager improperly marked the agent as disconnected'
    assert out_assign.hit_is_expired == True, \
        'Expired HIT was not marked as such'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_1), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_2), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    assert not mturk_manager.socket_manager.socket_is_open(connection_id_3), \
        'The socket manager didn\'t close the socket upon completion of the ' \
        'task, though it should have'
    completed_threads[UNIQUE_CONVERSATION_TEST] = True


# Map of tests to run to their testing function, slowest tests first reduces
# overall runtime
TESTS = {
    DUO_NO_ONBOARDING_TEST: test_duo_no_onboarding,
    SOLO_ONBOARDING_TEST: test_solo_with_onboarding,
    DUO_ONBOARDING_TEST: test_duo_with_onboarding,
    EXPIRE_HIT_TEST: test_expire_hit,
    DUO_ONE_DISCONNECT_TEST: test_duo_one_disconnect,
    DUO_VALID_RECONNECT_TEST: test_duo_valid_reconnects,
    UNIQUE_CONVERSATION_TEST: test_unique_workers_in_conversation,
    ALLOWED_CONVERSATION_TEST: test_allowed_conversations,
    SOLO_REFRESH_TEST: test_solo_refresh_in_middle,
    SOLO_NO_ONBOARDING_TEST: test_solo_no_onboarding,
    COUNT_COMPLETE_TEST: test_count_complete,
    SOCKET_TEST: test_socket_manager
}

# Runtime threads, MAX_THREADS is used on initial pass, RETEST_THREADS is used
# with flakey tests that failed under heavy load and thus may not have met
# the expected times for updating state
MAX_THREADS = 8
RETEST_THREADS = 2

def run_tests(tests_to_run, max_threads, base_opt, server_url):
    global start_time
    failed_tests = []
    threads = {}
    for test_name in tests_to_run:
        while len(threads) >= max_threads:
            new_threads = {}
            for n in threads:
                if threads[n].isAlive():
                    new_threads[n] = threads[n]
                else:
                    if n in completed_threads:
                        print("{} Passed. Runtime - {} Seconds".format(
                            n,
                            time.time() - start_times[n]
                        ))
                    else:
                        print("{} Failed. Runtime - {} Seconds".format(
                            n,
                            time.time() - start_times[n]
                        ))
                        failed_tests.append(n)
            threads = new_threads
            time.sleep(1)
        new_thread = threading.Thread(target=TESTS[test_name],
                                      args=(base_opt.copy(), server_url))
        new_thread.start()
        start_times[test_name] = time.time()
        threads[test_name] = new_thread
        time.sleep(0.25)
    while len(threads) > 0:
        new_threads = {}
        for n in threads:
            if threads[n].isAlive():
                new_threads[n] = threads[n]
            else:
                if n in completed_threads:
                    print("{} Passed. Runtime - {} Seconds".format(
                        n,
                        time.time() - start_times[n]
                    ))
                else:
                    print("{} Failed. Runtime - {} Seconds".format(
                        n,
                        time.time() - start_times[n]
                    ))
                    failed_tests.append(n)
        threads = new_threads
        time.sleep(1)
    return failed_tests


def main():
    start_time = time.time()
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    base_opt = argparser.parse_args()
    base_opt['is_sandbox'] = True
    base_opt['num_conversations'] = 1
    base_opt['count_complete'] = False
    task_name, server_url = handle_setup(base_opt)
    print ("Setup time: {} seconds".format(time.time() - start_time))
    start_time = time.time()
    try:
        failed_tests = run_tests(TESTS, MAX_THREADS, base_opt, server_url)
        if len(failed_tests) == 0:
            print("All tests passed, ParlAI MTurk is functioning")
        else:
            print("Some tests failed: ", failed_tests)
            print("Retrying flakey tests with fewer threads")
            flakey_tests = {}
            for test_name in failed_tests:
                flakey_tests[test_name] = TESTS[test_name]
            failed_tests = run_tests(flakey_tests, RETEST_THREADS, \
                                     base_opt, server_url)
            if len(failed_tests) == 0:
                print("All tests passed, ParlAI MTurk is functioning")
            else:
                print("Some tests failed even on retry: ", failed_tests)

        test_duration = time.time() - start_time
        print("Test duration: {} seconds".format(test_duration))
    except:
        raise
    finally:
        handle_shutdown(task_name)

if __name__ == '__main__':
    main()
