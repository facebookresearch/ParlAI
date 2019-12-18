#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import errno
import logging
import json
import threading
import time
from queue import PriorityQueue, Empty
import websocket

from parlai.mturk.core.dev.shared_utils import print_and_log
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils


class Packet:
    """
    Class for holding information sent over a socket.
    """

    # Possible Packet Status
    STATUS_NONE = -1
    STATUS_INIT = 0
    STATUS_SENT = 1
    STATUS_FAIL = 2

    # TODO remove unused attributes
    def __init__(
        self,
        id,
        type,
        sender_id,
        receiver_id,
        assignment_id,
        data,
        conversation_id=None,
        requires_ack=None,
        blocking=None,
        ack_func=None,
    ):
        """
        Create a packet to be used for holding information before it is
        sent through the socket
        id:               Unique ID to distinguish this packet from others
        type:             TYPE of packet (ACK, ALIVE, MESSAGE)
        sender_id:        Sender ID for this packet
        receiver_id:      Recipient ID for this packet
        assignment_id:    Assignment ID for this packet
        data:             Contents of the packet
        conversation_id:  Packet metadata - what conversation this belongs to
        requires_ack:     No longer used.
        blocking:         No longer used.
        ack_func:         Function to call upon successful ack of a packet
                           Default calls no function on ack
        """
        self.id = id
        # Possible Packet Types are set by data_model
        self.type = type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.assignment_id = assignment_id
        self.data = data
        self.conversation_id = conversation_id
        self.ack_func = ack_func
        self.status = self.STATUS_INIT

    @staticmethod
    def from_dict(packet):
        """
        Create a packet from the dictionary that would be recieved over a socket.
        """
        try:
            packet_id = packet['id']
            packet_type = packet['type']
            sender_id = packet['sender_id']
            receiver_id = packet.get('receiver_id', None)
            assignment_id = packet.get('assignment_id', None)
            data = packet.get('data', '')
            conversation_id = packet.get('conversation_id', None)

            return Packet(
                packet_id,
                packet_type,
                sender_id,
                receiver_id,
                assignment_id,
                data,
                conversation_id,
            )
        except Exception as e:
            print_and_log(
                logging.WARN,
                'Could not create a valid packet out of the dictionary'
                'provided: {}, error: {}'.format(packet, repr(e)),
            )
            return None

    def as_dict(self):
        """
        Convert a packet into a form that can be pushed over a socket.
        """
        return {
            'id': self.id,
            'type': self.type,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'assignment_id': self.assignment_id,
            'conversation_id': self.conversation_id,
            'data': self.data,
        }

    def get_sender_connection_id(self):
        """
        Get the connection_id that this packet came from.
        """
        return '{}_{}'.format(self.sender_id, self.assignment_id)

    def get_receiver_connection_id(self):
        """
        Get the connection_id that this is going to.
        """
        return '{}_{}'.format(self.receiver_id, self.assignment_id)

    def new_copy(self):
        """
        Return a new packet that is a copy of this packet with a new id and with a fresh
        status.
        """
        packet = Packet.from_dict(self.as_dict())
        packet.id = shared_utils.generate_event_id(self.receiver_id)
        return packet

    def __repr__(self):
        return 'Packet <{}>'.format(self.as_dict())

    def swap_sender(self):
        """
        Swaps the sender_id and receiver_id.
        """
        self.sender_id, self.receiver_id = self.receiver_id, self.sender_id
        return self

    def set_type(self, new_type):
        """
        Updates the message type.
        """
        self.type = new_type
        return self

    def set_data(self, new_data):
        """
        Updates the message data.
        """
        self.data = new_data
        return self


class SocketManager:
    """
    SocketManager is a wrapper around websocket to conform to the API allowing the
    remote state to sync up with our local state.
    """

    # Default pings without pong before socket considered dead
    DEF_MISSED_PONGS: int = 20
    PING_RATE: float = 4
    DEF_DEAD_TIME: float = 30

    def __init__(
        self,
        server_url,
        port,
        alive_callback,
        message_callback,
        socket_dead_callback,
        task_group_id,
        socket_dead_timeout=None,
        server_death_callback=None,
    ):
        """
        server_url:           url at which the server is to be run
        port:                 port for the socket to operate on
        alive_callback:       function to be called on alive Packets, defined
                               alive_callback(self, pkt)
        message_callback:     function to be called on message Packets, defined
                               message_callback(self, pkt)
        socket_dead_callback: function to be called when a socket dies, should
                              return false if the socket_manager should ignore
                              the death and treat the socket as alive defined
                               on_socket_dead(self, worker_id, assignment_id)
        socket_dead_timeout:  time to wait between pings before dying
        """
        self.server_url = server_url
        self.port = port
        self.alive_callback = alive_callback
        self.message_callback = message_callback
        self.socket_dead_callback = socket_dead_callback
        self.server_death_callback = server_death_callback
        if socket_dead_timeout is not None:
            self.missed_pongs = 1 + socket_dead_timeout / self.PING_RATE
        else:
            self.missed_pongs = self.DEF_MISSED_PONGS
        self.task_group_id = task_group_id

        self.ws = None
        self.keep_running = True

        # initialize the state
        self.listen_thread = None
        self.send_thread = None
        self.sending_queue = PriorityQueue()
        self.open_channels = set()
        self.last_sent_ping_time = 0  # time of last ping send
        self.pings_without_pong = 0
        self.processed_packets = set()
        self.packet_map = {}
        self.alive = False
        self.is_shutdown = False
        self.send_lock = threading.Condition()
        self.packet_map_lock = threading.Condition()
        self.worker_assign_ids = {}  # mapping from connection id to pair

        # setup the socket
        self._setup_socket()

    def get_my_sender_id(self):
        """
        Gives the name that this socket manager should use for its world.
        """
        return '[World_{}]'.format(self.task_group_id)

    def _safe_send(self, data, force=False):
        if not self.alive and not force:
            # Try to wait a half second to send a packet
            timeout = 0.5
            while timeout > 0 and not self.alive:
                time.sleep(0.1)
                timeout -= 0.1
            if not self.alive:
                # don't try to send a packet if we're still dead
                return False
        try:
            with self.send_lock:
                self.ws.send(data)
        except websocket.WebSocketConnectionClosedException:
            # The channel died mid-send, wait for it to come back up
            return False
        except BrokenPipeError:  # noqa F821 we don't support p2
            # The channel died mid-send, wait for it to come back up
            return False
        except AttributeError:
            # _ensure_closed was called in parallel, self.ws = None
            return False
        except Exception as e:
            shared_utils.print_and_log(
                logging.WARN,
                'Unexpected socket error occured: {}'.format(repr(e)),
                should_print=True,
            )
            return False
        return True

    def _ensure_closed(self):
        self.alive = False
        if self.ws is None:
            return
        try:
            self.ws.close()
        except websocket.WebSocketConnectionClosedException:
            pass
        self.ws = None

    def _send_world_alive(self):
        """
        Registers world with the passthrough server.
        """
        self._safe_send(
            json.dumps(
                {
                    'type': data_model.AGENT_ALIVE,
                    'content': {
                        'id': 'WORLD_ALIVE',
                        'sender_id': self.get_my_sender_id(),
                    },
                }
            ),
            force=True,
        )

    def _try_send_world_ping(self):
        if time.time() - self.last_sent_ping_time > self.PING_RATE:
            self._safe_send(
                json.dumps(
                    {
                        'type': data_model.WORLD_PING,
                        'content': {
                            'id': 'WORLD_PING',
                            'sender_id': self.get_my_sender_id(),
                        },
                    }
                ),
                force=True,
            )
            self.last_sent_ping_time = time.time()

    def _send_packet(self, packet, send_time):
        """
        Sends a packet, blocks if the packet is blocking.
        """
        # Send the packet
        pkt = packet.as_dict()
        if pkt['data'] is None or packet.status == Packet.STATUS_SENT:
            return  # This packet was _just_ sent.
        shared_utils.print_and_log(logging.DEBUG, 'Send packet: {}'.format(packet))

        result = self._safe_send(json.dumps({'type': pkt['type'], 'content': pkt}))
        if not result:
            # The channel died mid-send, wait for it to come back up
            self.sending_queue.put((send_time, packet))
            return

        packet.status = Packet.STATUS_SENT
        if packet.ack_func is not None:
            packet.ack_func(packet)

    def _spawn_reaper_thread(self):
        def _reaper_thread(*args):
            start_time = time.time()
            wait_time = self.DEF_MISSED_PONGS * self.PING_RATE
            while time.time() - start_time < wait_time:
                if self.is_shutdown:
                    return
                if self.alive:
                    return
                time.sleep(0.3)
            if self.server_death_callback is not None:
                shared_utils.print_and_log(
                    logging.WARN,
                    'Server has disconnected and could not reconnect. '
                    'Assuming the worst and calling the death callback. '
                    '(Usually shutdown)',
                    should_print=True,
                )
                self.server_death_callback()

        reaper_thread = threading.Thread(
            target=_reaper_thread, name='socket-reaper-{}'.format(self.task_group_id)
        )
        reaper_thread.daemon = True
        reaper_thread.start()

    def _setup_socket(self):
        """
        Create socket handlers and registers the socket.
        """

        def on_socket_open(*args):
            shared_utils.print_and_log(logging.DEBUG, 'Socket open: {}'.format(args))
            self._send_world_alive()

        def on_error(ws, error):
            try:
                if error.errno == errno.ECONNREFUSED:
                    self.use_socket = False
                    self._ensure_closed()
                    raise Exception("Socket refused connection, cancelling")
                else:
                    shared_utils.print_and_log(
                        logging.WARN, 'Socket logged error: {}'.format(error)
                    )
                    self._ensure_closed()
            except Exception:
                if type(error) is websocket.WebSocketConnectionClosedException:
                    return  # Connection closed is noop
                shared_utils.print_and_log(
                    logging.WARN,
                    'Socket logged error: {} Restarting'.format(repr(error)),
                )
                self._ensure_closed()

        def on_disconnect(*args):
            """
            Disconnect event is a no-op for us, as the server reconnects automatically
            on a retry.

            Just in case the server is actually dead we set up a thread to reap the
            whole task.
            """
            shared_utils.print_and_log(
                logging.INFO, 'World server disconnected: {}'.format(args)
            )
            self._ensure_closed()
            if not self.is_shutdown:
                self._spawn_reaper_thread()

        def on_message(*args):
            """
            Incoming message handler for SERVER_PONG, MESSAGE_BATCH, AGENT_DISCONNECT,
            SNS_MESSAGE, SUBMIT_MESSAGE, AGENT_ALIVE.
            """
            packet_dict = json.loads(args[1])
            if packet_dict['type'] == 'conn_success':  # TODO make socket func
                self.alive = True
                return
            # The packet inherits the socket function type
            packet_dict['content']['type'] = packet_dict['type']
            packet = Packet.from_dict(packet_dict['content'])
            if packet is None:
                return
            packet_id = packet.id
            packet_type = packet.type
            if packet_id in self.processed_packets:
                return  # no need to handle already-processed packets

            # Note to self that this packet has already been processed,
            # and shouldn't be processed again in the future
            self.processed_packets.add(packet_id)
            if packet_type == data_model.SERVER_PONG:
                # Incoming pong means our ping was returned
                self.pings_without_pong = 0
            elif packet_type == data_model.AGENT_ALIVE:
                # agent is connecting for the first time
                self.alive_callback(packet)
                self.processed_packets.add(packet_id)
            elif packet_type == data_model.MESSAGE_BATCH:
                # Any number of agents are included in this message batch,
                # so process each individually
                batched_packets = packet.data['messages']
                for batched_packet_dict in batched_packets:
                    batched_packet_dict['type'] = data_model.AGENT_MESSAGE
                    batched_packet = Packet.from_dict(batched_packet_dict)
                    self.message_callback(batched_packet)
            elif packet_type == data_model.AGENT_DISCONNECT:
                # Server detected an agent disconnect, extract and remove
                disconnected_id = packet.data['connection_id']
                worker_id, assign_id = self.worker_assign_ids[disconnected_id]
                self.socket_dead_callback(worker_id, assign_id)
            elif packet_type == data_model.SNS_MESSAGE:
                # Treated as a regular message
                self.message_callback(packet)
            elif packet_type == data_model.SUBMIT_MESSAGE:
                # Treated as a regular message
                self.message_callback(packet)

        def run_socket(*args):
            url_base_name = self.server_url.split('https://')[1]
            protocol = "wss"
            if url_base_name in ['localhost', '127.0.0.1']:
                protocol = "ws"
            while self.keep_running:
                try:
                    sock_addr = "{}://{}:{}/".format(protocol, url_base_name, self.port)
                    self.ws = websocket.WebSocketApp(
                        sock_addr,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_disconnect,
                    )
                    self.ws.on_open = on_socket_open
                    self.ws.run_forever(ping_interval=8 * self.PING_RATE)
                    self._ensure_closed()
                except Exception as e:
                    shared_utils.print_and_log(
                        logging.WARN,
                        'Socket error {}, attempting restart'.format(repr(e)),
                    )
                time.sleep(0.2)

        # Start listening thread
        self.listen_thread = threading.Thread(
            target=run_socket, name='Main-Socket-Recv-Thread'
        )
        self.listen_thread.daemon = True
        self.listen_thread.start()
        time.sleep(1.2)
        start_time = time.time()
        while not self.alive:
            if time.time() - start_time > self.DEF_DEAD_TIME:
                self.server_death_callback()
                raise ConnectionRefusedError(  # noqa F821 we only support py3
                    'Was not able to establish a connection with the server, '
                    'please try to run again. If that fails,'
                    'please ensure that your local device has the correct SSL '
                    'certs installed.'
                )
            try:
                self._send_world_alive()
            except Exception:
                pass
            time.sleep(self.PING_RATE / 2)

        # Start sending thread
        self.send_thread = threading.Thread(
            target=self.channel_thread, name='Main-Socket-Send-Thread'
        )
        self.send_thread.daemon = True
        self.send_thread.start()

    def channel_thread(self):
        """
        Handler thread for monitoring all channels.
        """
        # while the thread is still alive
        while not self.is_shutdown:
            if self.ws is None:
                # Waiting for websocket to come back alive
                time.sleep(shared_utils.THREAD_SHORT_SLEEP)
                continue
            self._try_send_world_ping()
            try:
                # Get first item in the queue, check if can send it yet
                item = self.sending_queue.get(block=False)
                t = item[0]
                if time.time() < t:
                    # Put the item back into the queue,
                    # it's not time to pop yet
                    self.sending_queue.put(item)
                else:
                    # Try to send the packet
                    packet = item[1]
                    if not packet:
                        # This packet was deleted out from under us
                        continue
                    if packet.status is not Packet.STATUS_SENT:
                        # either need to send initial packet
                        # or resend after a failed send
                        self._send_packet(packet, t)
            except Empty:
                time.sleep(shared_utils.THREAD_SHORT_SLEEP)
            except Exception as e:
                shared_utils.print_and_log(
                    logging.WARN,
                    'Unexpected error occurred in socket handling thread: '
                    '{}'.format(repr(e)),
                    should_print=True,
                )

    # Inividual channel accessors are useful for testing
    def open_channel(self, worker_id, assignment_id):
        """
        Opens a channel for a worker on a given assignment, doesn't re-open if the
        channel is already open.
        """
        connection_id = '{}_{}'.format(worker_id, assignment_id)
        self.open_channels.add(connection_id)
        self.worker_assign_ids[connection_id] = (worker_id, assignment_id)

    def close_channel(self, connection_id):
        """
        Closes a channel by connection_id.
        """
        shared_utils.print_and_log(
            logging.DEBUG, 'Closing channel {}'.format(connection_id)
        )
        if connection_id in self.open_channels:
            self.open_channels.remove(connection_id)
            with self.packet_map_lock:
                packet_ids = list(self.packet_map.keys())
                # Clear packets associated with this sender
                for packet_id in packet_ids:
                    packet = self.packet_map[packet_id]
                    packet_conn_id = packet.get_receiver_connection_id()
                    if connection_id == packet_conn_id:
                        del self.packet_map[packet_id]

    def close_all_channels(self):
        """
        Closes all channels by clearing the list of channels.
        """
        shared_utils.print_and_log(logging.DEBUG, 'Closing all channels')
        connection_ids = list(self.open_channels)
        for connection_id in connection_ids:
            self.close_channel(connection_id)

    def socket_is_open(self, connection_id):
        return connection_id in self.open_channels

    def queue_packet(self, packet):
        """
        Queues sending a packet to its intended owner.
        """
        shared_utils.print_and_log(
            logging.DEBUG, 'Put packet ({}) in queue'.format(packet.id)
        )
        # Get the current time to put packet into the priority queue
        with self.packet_map_lock:
            self.packet_map[packet.id] = packet
        item = (time.time(), packet)
        self.sending_queue.put(item)
        return True

    def get_status(self, packet_id):
        """
        Returns the status of a particular packet by id.
        """
        with self.packet_map_lock:
            if packet_id not in self.packet_map:
                return Packet.STATUS_NONE
            return self.packet_map[packet_id].status

    def shutdown(self):
        """
        marks the socket manager as closing, shuts down all channels.
        """
        self.is_shutdown = True
        self.close_all_channels()
        self.keep_running = False
        self._ensure_closed()
