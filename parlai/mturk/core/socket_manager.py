# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import threading
import time
from queue import PriorityQueue, Empty
from socketIO_client_nexus import SocketIO
from parlai.mturk.core.shared_utils import print_and_log, generate_event_id, \
                                        THREAD_SHORT_SLEEP, THREAD_MEDIUM_SLEEP
import parlai.mturk.core.data_model as data_model

class Packet():
    """Class for holding information sent over a socket"""

    # Possible Packet Status
    STATUS_INIT = 0
    STATUS_SENT = 1
    STATUS_ACK = 2

    # Possible Packet Types
    TYPE_ACK = 'ack'
    TYPE_ALIVE = 'alive'
    TYPE_MESSAGE = 'message'
    TYPE_HEARTBEAT = 'heartbeat'

    def __init__(self, id, type, sender_id, receiver_id, assignment_id, data,
                 conversation_id=None, requires_ack=True, blocking=True,
                 ack_func=None):
        """
        Create a packet to be used for holding information before it is
        sent through the socket
        id:               Unique ID to distinguish this packet from others
        type:             TYPE of packet (ACK, ALIVE, MESSAGE, HEARTBEAT)
        sender_id:        Sender ID for this packet
        receiver_id:      Recipient ID for this packet
        assignment_id:    Assignment ID for this packet
        data:             Contents of the packet
        conversation_id:  Packet metadata - what conversation this belongs to
        requires_ack:     Whether or not this packet needs to be acknowledged,
                           determines if retry logic will be used until ack is
                           recieved.
        blocking:         Whether or not this packet requires blocking to
                           remain in order amongst other packets in the queue.
                           Default is True
        ack_func:         Function to call upon successful ack of a packet
                           Default calls no function on ack
        """
        self.id = id
        self.type = type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.assignment_id = assignment_id
        self.data = data
        self.conversation_id = conversation_id
        self.requires_ack = requires_ack
        self.blocking = blocking
        self.ack_func = ack_func
        self.status = self.STATUS_INIT
        self.time = None

    @staticmethod
    def from_dict(packet):
        """Create a packet from the dictionary that would
        be recieved over a socket
        """
        packet_id = packet['id']
        packet_type = packet['type']
        sender_id = packet['sender_id']
        receiver_id = packet['receiver_id']
        assignment_id = packet['assignment_id']
        data = None
        if 'data' in packet:
            data = packet['data']
        else:
            data = ''
        conversation_id = packet['conversation_id']

        return Packet(packet_id, packet_type, sender_id, receiver_id,
            assignment_id, data, conversation_id)

    def as_dict(self):
        """Convert a packet into a form that can be pushed over a socket"""
        return {
            'id': self.id,
            'type': self.type,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'assignment_id': self.assignment_id,
            'conversation_id': self.conversation_id,
            'data': self.data
        }

    def get_sender_connection_id(self):
        """Get the connection_id that this packet came from"""
        return '{}_{}'.format(self.sender_id, self.assignment_id)

    def get_receiver_connection_id(self):
        """Get the connection_id that this is going to"""
        return '{}_{}'.format(self.receiver_id, self.assignment_id)

    def get_ack(self):
        """Return a new packet that can be used to acknowledge this packet"""
        return Packet(self.id, self.TYPE_ACK, self.receiver_id, self.sender_id,
            self.assignment_id, '', self.conversation_id, False, False)

    def new_copy(self):
        """Return a new packet that is a copy of this packet with
        a new id and with a fresh status
        """
        packet = Packet.from_dict(self.as_dict())
        packet.id = generate_event_id(self.receiver_id)
        return packet

    def __repr__(self):
        return 'Packet <{}>'.format(self.as_dict())

    def swap_sender(self):
        """Swaps the sender_id and receiver_id"""
        self.sender_id, self.receiver_id = self.receiver_id, self.sender_id
        return self

    def set_type(self, new_type):
        """Updates the message type"""
        self.type = new_type
        return self

    def set_data(self, new_data):
        """Updates the message data"""
        self.data = new_data
        return self


class SocketManager():
    """SocketManager is a wrapper around socketIO to stabilize its packet
    passing. The manager handles resending packet, as well as maintaining
    alive status for all the connections it forms
    """

    # Time to acknowledge different message types
    ACK_TIME = {Packet.TYPE_ALIVE: 2,
                Packet.TYPE_MESSAGE: 2}

    # Default time before socket deemed dead
    DEF_SOCKET_TIMEOUT = 8

    def __init__(self, server_url, port, alive_callback, message_callback,
                 socket_dead_callback, task_group_id,
                 socket_dead_timeout=None):
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
        socket_dead_timeout:  time to wait between heartbeats before dying
        """
        self.server_url = server_url
        self.port = port
        self.alive_callback = alive_callback
        self.message_callback = message_callback
        self.socket_dead_callback = socket_dead_callback
        if socket_dead_timeout == None:
            self.socket_dead_timeout = self.DEF_SOCKET_TIMEOUT
        else:
            self.socket_dead_timeout = socket_dead_timeout
        self.task_group_id = task_group_id

        self.socketIO = None

        # initialize the state
        self.listen_thread = None
        self.queues = {}
        self.threads = {}
        self.run = {}
        self.last_heartbeat = {}
        self.packet_map = {}

        # setup the socket
        self._setup_socket()

    def get_my_sender_id(self):
        """Gives the name that this socket manager should use for its world"""
        return '[World_{}]'.format(self.task_group_id)

    def _send_world_alive(self):
        """Registers world with the passthrough server"""
        self.socketIO.emit(
            data_model.SOCKET_AGENT_ALIVE_STRING,
            {'id': 'WORLD_ALIVE', 'sender_id': self.get_my_sender_id()}
        )

    def _send_response_heartbeat(self, packet):
        """Sends a response heartbeat to an incoming heartbeat packet"""
        self.socketIO.emit(
            data_model.SOCKET_ROUTE_PACKET_STRING,
            packet.swap_sender().set_data('').as_dict()
        )

    def _send_ack(self, packet):
        """Sends an ack to a given packet"""
        ack = packet.get_ack().as_dict()
        self.socketIO.emit(data_model.SOCKET_ROUTE_PACKET_STRING, ack, None)

    def _send_packet(self, packet, connection_id, send_time):
        """Sends a packet, blocks if the packet is blocking"""
        # Send the packet
        pkt = packet.as_dict()
        print_and_log('Send packet: {}'.format(packet.data))
        def set_status_to_sent(data):
            packet.status = Packet.STATUS_SENT
        self.socketIO.emit(
            data_model.SOCKET_ROUTE_PACKET_STRING,
            pkt,
            set_status_to_sent
        )

        # Handles acks and blocking
        if packet.requires_ack:
            if packet.blocking:
                # blocking till ack is received or timeout
                start_t = time.time()
                while True:
                    if packet.status == Packet.STATUS_ACK:
                        break
                    if time.time() - start_t > self.ACK_TIME[packet.type]:
                        # didn't receive ACK, resend packet keep old queue time
                        # to ensure this packet is processed first
                        packet.status = Packet.STATUS_INIT
                        self.queues[connection_id].put((send_time, packet))
                        break
                    time.sleep(THREAD_SHORT_SLEEP)
            else:
                # non-blocking ack: add ack-check to queue
                t = time.time() + self.ACK_TIME[packet.type]
                self.queues[connection_id].put((t, packet))

    def _setup_socket(self):
        """Create socket handlers and registers the socket"""
        self.socketIO = SocketIO(self.server_url, self.port)

        def on_socket_open(*args):
            print_and_log('Socket open: {}'.format(args), False)
            self._send_world_alive()

        def on_disconnect(*args):
            """Disconnect event is a no-op for us, as the server reconnects
            automatically on a retry"""
            print_and_log('World server disconnected: {}'.format(args), False)

        def on_message(*args):
            """Incoming message handler for ACKs, ALIVEs, HEARTBEATs,
            and MESSAGEs"""
            packet = Packet.from_dict(args[0])
            packet_id = packet.id
            packet_type = packet.type
            connection_id = packet.get_sender_connection_id()
            if packet_type == Packet.TYPE_ACK:
                # Acknowledgements should mark a packet as acknowledged
                print_and_log('On new ack: {}'.format(args), False)
                self.packet_map[packet_id].status = Packet.STATUS_ACK
                # If the packet sender wanted to do something on acknowledge
                if self.packet_map[packet_id].ack_func:
                    self.packet_map[packet_id].ack_func(packet)
            elif packet_type == Packet.TYPE_HEARTBEAT:
                # Heartbeats update the last heartbeat time and respond in kind
                self.last_heartbeat[connection_id] = time.time()
                self._send_response_heartbeat(packet)
            else:
                # Remaining packet types need to be acknowledged
                print_and_log('On new message: {}'.format(args), False)
                self._send_ack(packet)
                # Call the appropriate callback
                if packet_type == Packet.TYPE_ALIVE:
                    self.last_heartbeat[connection_id] = time.time()
                    self.alive_callback(packet)
                elif packet_type == Packet.TYPE_MESSAGE:
                    self.message_callback(packet)

        # Register Handlers
        self.socketIO.on(data_model.SOCKET_OPEN_STRING, on_socket_open)
        self.socketIO.on(data_model.SOCKET_DISCONNECT_STRING, on_disconnect)
        self.socketIO.on(data_model.SOCKET_NEW_PACKET_STRING, on_message)

        # Start listening thread
        self.listen_thread = threading.Thread(target=self.socketIO.wait)
        self.listen_thread.daemon = True
        self.listen_thread.start()

    def open_channel(self, worker_id, assignment_id):
        """Opens a channel for a worker on a given assignment, doesn't re-open
        if the channel is already open. Handles creation of the thread that
        monitors that channel"""
        connection_id = '{}_{}'.format(worker_id, assignment_id)
        if connection_id in self.queues and self.run[connection_id]:
            print_and_log(
                'Channel ({}) already open'.format(connection_id),
                False
            )
            return
        self.run[connection_id] = True
        self.queues[connection_id] = PriorityQueue()

        def channel_thread():
            """Handler thread for monitoring a single channel"""
            # while the thread is still alive
            while self.run[connection_id]:
                try:
                    # Check if client is still alive
                    if (time.time() - self.last_heartbeat[connection_id]
                            > self.socket_dead_timeout):
                        if self.socket_dead_callback(worker_id, assignment_id):
                            self.run[connection_id] = False

                    # Make sure the queue still exists
                        if not connection_id in self.queues:
                            self.run[connection_id] = False
                            break

                    # Get first item in the queue, check if we can send it yet
                    item = self.queues[connection_id].get(block=False)
                    t = item[0]
                    if time.time() < t:
                        # Put the item back into the queue,
                        # it's not time to pop yet
                        self.queues[connection_id].put(item)
                    else:
                        # Try to send the packet
                        packet = item[1]
                        if packet.status is not Packet.STATUS_ACK:
                            # either need to send initial packet
                            # or resend not-acked packet
                            self._send_packet(packet, connection_id, t)
                except Empty:
                    pass
                finally:
                    time.sleep(THREAD_MEDIUM_SLEEP)

        # Setup and run the channel sending thread
        self.threads[connection_id] = threading.Thread(target=channel_thread)
        self.threads[connection_id].daemon = True
        self.threads[connection_id].start()

    def _close_channel_internal(self, connection_id):
        """Closes a channel by connection_id"""
        print_and_log('Closing channel {}'.format(connection_id), False)
        self.run[connection_id] = False
        if connection_id in self.queues:
            del self.queues[connection_id]
            del self.threads[connection_id]

    def close_channel(self, worker_id, assignment_id):
        """Closes a channel by worker_id and assignment_id"""
        self._close_channel_internal('{}_{}'.format(worker_id, assignment_id))

    def close_all_channels(self):
        """Closes a channel by clearing the list of channels"""
        print_and_log('Closing all channels')
        connection_ids = list(self.queues.keys())
        for connection_id in connection_ids:
            self._close_channel_internal(connection_id)

    def socket_is_open(self, connection_id):
        return connection_id in self.queues

    def queue_packet(self, packet):
        """Queues sending a packet to its intended owner"""
        connection_id = packet.get_receiver_connection_id()
        if not self.socket_is_open(connection_id):
            # Warn if there is no socket to send through for the expected recip
            print_and_log(
                'Can not send packet to worker_id {}: packet queue not found. '
                'Message: {}'.format(connection_id, packet.data)
            )
            return
        print_and_log('Put packet ({}) in queue ({})'.format(
            packet.id,
            connection_id
        ), False)
        # Get the current time to put packet into the priority queue
        self.packet_map[packet.id] = packet
        item = (time.time(), packet)
        self.queues[connection_id].put(item)

    def get_status(self, packet_id):
        """Returns the status of a particular packet by id"""
        return self.packet_map[packet_id].status
