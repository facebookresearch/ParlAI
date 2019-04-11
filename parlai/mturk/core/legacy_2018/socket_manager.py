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

from parlai.mturk.core.shared_utils import print_and_log
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils


class Packet():
    """Class for holding information sent over a socket"""

    # Possible Packet Status
    STATUS_NONE = -1
    STATUS_INIT = 0
    STATUS_SENT = 1
    STATUS_ACK = 2
    STATUS_FAIL = 3

    # Possible Packet Types
    TYPE_ACK = 'ack'
    TYPE_ALIVE = 'alive'
    TYPE_MESSAGE = 'message'
    TYPE_HEARTBEAT = 'heartbeat'
    TYPE_PONG = 'pong'

    def __init__(self, id, type, sender_id, receiver_id, assignment_id, data,
                 conversation_id=None, requires_ack=None, blocking=None,
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
                           recieved. Default for messages and alives is True,
                           default for the rest is false.
        blocking:         Whether or not this packet requires blocking to
                           remain in order amongst other packets in the queue.
                           Default is for messages and alives is True, default
                           for the rest is false.
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
        important = self.type in [self.TYPE_ALIVE, self.TYPE_MESSAGE]
        self.requires_ack = important if requires_ack is None else requires_ack
        self.blocking = important if blocking is None else blocking
        self.ack_func = ack_func
        self.status = self.STATUS_INIT

    @staticmethod
    def from_dict(packet):
        """Create a packet from the dictionary that would
        be recieved over a socket
        """
        try:
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
        except Exception:
            print_and_log(
                logging.WARN,
                'Could not create a valid packet out of the dictionary'
                'provided: {}'.format(packet)
            )
            return None

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
                      self.assignment_id, '', self.conversation_id, False,
                      False)

    def new_copy(self):
        """Return a new packet that is a copy of this packet with
        a new id and with a fresh status
        """
        packet = Packet.from_dict(self.as_dict())
        packet.id = shared_utils.generate_event_id(self.receiver_id)
        packet.blocking = self.blocking
        packet.requires_ack = self.requires_ack
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
    """SocketManager is a wrapper around websocket to stabilize its packet
    passing. The manager handles resending packet, as well as maintaining
    alive status for all the connections it forms
    """

    # Time to acknowledge different message types
    ACK_TIME = {Packet.TYPE_ALIVE: 4,
                Packet.TYPE_MESSAGE: 4}

    # Default pongs without heartbeat before socket considered dead
    DEF_MISSED_PONGS = 20
    HEARTBEAT_RATE = 4
    DEF_DEAD_TIME = 30

    def __init__(self, server_url, port, alive_callback, message_callback,
                 socket_dead_callback, task_group_id,
                 socket_dead_timeout=None, server_death_callback=None):
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
        self.server_death_callback = server_death_callback
        if socket_dead_timeout is not None:
            self.missed_pongs = 1 + socket_dead_timeout / self.HEARTBEAT_RATE
        else:
            self.missed_pongs = self.DEF_MISSED_PONGS
        self.task_group_id = task_group_id

        self.ws = None
        self.keep_running = True

        # initialize the state
        self.listen_thread = None
        self.send_thread = None
        self.queues = {}
        self.blocking_packets = {}  # connection_id to blocking packet map
        self.threads = {}
        self.run = {}
        self.last_sent_heartbeat_time = {}  # time of last heartbeat sent
        self.last_received_heartbeat = {}  # actual last received heartbeat
        self.pongs_without_heartbeat = {}
        # TODO update processed packets to *ensure* only one execution per
        # packet, as right now two threads can be spawned to process the
        # same packet at the same time, as the packet is only added to this
        # set after processing is complete.
        self.processed_packets = set()
        self.packet_map = {}
        self.alive = False
        self.is_shutdown = False
        self.send_lock = threading.Condition()
        self.packet_map_lock = threading.Condition()
        self.thread_shutdown_lock = threading.Condition()
        self.worker_assign_ids = {}  # mapping from connection id to pair

        # setup the socket
        self._setup_socket()

    def get_my_sender_id(self):
        """Gives the name that this socket manager should use for its world"""
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
                should_print=True
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
        """Registers world with the passthrough server"""
        self._safe_send(json.dumps({
            'type': data_model.SOCKET_AGENT_ALIVE_STRING,
            'content':
                {'id': 'WORLD_ALIVE', 'sender_id': self.get_my_sender_id()},
        }), force=True)

    def _send_needed_heartbeat(self, connection_id):
        """Sends a heartbeat to a connection if needed"""
        if connection_id not in self.last_received_heartbeat:
            return
        if self.last_received_heartbeat[connection_id] is None:
            return
        if (time.time() - self.last_sent_heartbeat_time[connection_id] <
                self.HEARTBEAT_RATE):
            return
        packet = self.last_received_heartbeat[connection_id]
        self._safe_send(json.dumps({
            'type': data_model.SOCKET_ROUTE_PACKET_STRING,
            'content': packet.new_copy().swap_sender().set_data('').as_dict()
        }))
        self.last_sent_heartbeat_time[connection_id] = time.time()

    def _send_ack(self, packet):
        """Sends an ack to a given packet"""
        ack = packet.get_ack().as_dict()
        result = self._safe_send(json.dumps({
            'type': data_model.SOCKET_ROUTE_PACKET_STRING,
            'content': ack,
        }))
        if result:
            packet.status = Packet.STATUS_SENT

    def _send_packet(self, packet, connection_id, send_time):
        """Sends a packet, blocks if the packet is blocking"""
        # Send the packet
        pkt = packet.as_dict()
        if pkt['data'] is None or packet.status == Packet.STATUS_ACK:
            return  # This packet was _just_ acked.
        shared_utils.print_and_log(
            logging.DEBUG,
            'Send packet: {}'.format(packet)
        )

        result = self._safe_send(json.dumps({
            'type': data_model.SOCKET_ROUTE_PACKET_STRING,
            'content': pkt,
        }))
        if not result:
            # The channel died mid-send, wait for it to come back up
            self._safe_put(connection_id, (send_time, packet))
            return

        if packet.status != Packet.STATUS_ACK:
            packet.status = Packet.STATUS_SENT

        # Handles acks and blocking
        if packet.requires_ack:
            if packet.blocking:
                # Put the packet right back into its place to prevent sending
                # other packets, then block that connection
                self._safe_put(connection_id, (send_time, packet))
                t = time.time() + self.ACK_TIME[packet.type]
                self.blocking_packets[connection_id] = (t, packet)
            else:
                # non-blocking ack: add ack-check to queue
                t = time.time() + self.ACK_TIME[packet.type]
                self._safe_put(connection_id, (t, packet))

    def _spawn_reaper_thread(self):
        def _reaper_thread(*args):
            start_time = time.time()
            wait_time = self.DEF_MISSED_PONGS * self.HEARTBEAT_RATE
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
            target=_reaper_thread,
            name='socket-reaper-{}'.format(self.task_group_id)
        )
        reaper_thread.daemon = True
        reaper_thread.start()

    def _setup_socket(self):
        """Create socket handlers and registers the socket"""
        def on_socket_open(*args):
            shared_utils.print_and_log(
                logging.DEBUG,
                'Socket open: {}'.format(args)
            )
            self._send_world_alive()

        def on_error(ws, error):
            try:
                if error.errno == errno.ECONNREFUSED:
                    self.use_socket = False
                    self._ensure_closed()
                    raise Exception("Socket refused connection, cancelling")
                else:
                    shared_utils.print_and_log(
                        logging.WARN,
                        'Socket logged error: {}'.format(error),
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
            """Disconnect event is a no-op for us, as the server reconnects
            automatically on a retry. Just in case the server is actually
            dead we set up a thread to reap the whole task.
            """
            shared_utils.print_and_log(
                logging.INFO,
                'World server disconnected: {}'.format(args)
            )
            self._ensure_closed()
            if not self.is_shutdown:
                self._spawn_reaper_thread()

        def on_message(*args):
            """Incoming message handler for ACKs, ALIVEs, HEARTBEATs,
            PONGs, and MESSAGEs"""
            packet_dict = json.loads(args[1])
            if packet_dict['type'] == 'conn_success':
                self.alive = True
                return  # No action for successful connection
            packet = Packet.from_dict(packet_dict['content'])
            if packet is None:
                return
            packet_id = packet.id
            packet_type = packet.type
            connection_id = packet.get_sender_connection_id()
            if packet_type == Packet.TYPE_ACK:
                with self.packet_map_lock:
                    if packet_id not in self.packet_map:
                        # Don't do anything when acking a packet we don't have
                        return
                    # Acknowledgements should mark a packet as acknowledged
                    shared_utils.print_and_log(
                        logging.DEBUG,
                        'On new ack: {}'.format(args)
                    )
                    self.packet_map[packet_id].status = Packet.STATUS_ACK
                    # If the packet sender wanted to do something on ack
                    if self.packet_map[packet_id].ack_func:
                        self.packet_map[packet_id].ack_func(packet)
                    # clear the stored packet data for memory reasons
                    try:
                        self.packet_map[packet_id].data = None
                    except Exception:
                        pass  # state already reduced, perhaps by ack_func
            elif packet_type == Packet.TYPE_HEARTBEAT:
                # Heartbeats update the last heartbeat, clears pongs w/o beat
                self.last_received_heartbeat[connection_id] = packet
                self.pongs_without_heartbeat[connection_id] = 0
            elif packet_type == Packet.TYPE_PONG:
                # Message in response from the router, ensuring we're connected
                # to it. Redundant but useful for metering from web client.
                pong_conn_id = packet.get_receiver_connection_id()
                if self.last_received_heartbeat[pong_conn_id] is not None:
                    self.pongs_without_heartbeat[pong_conn_id] += 1
            elif packet_id in self.processed_packets:
                # We don't want to re-process a packet that has already
                # been recieved, but we do need to tell the sender it is ack'd
                # re-acknowledge the packet was recieved and already processed
                self._send_ack(packet)
            else:

                # Remaining packet types need to be acknowledged
                shared_utils.print_and_log(
                    logging.DEBUG,
                    'On new message: {}'.format(args)
                )
                # Call the appropriate callback
                if packet_type == Packet.TYPE_ALIVE:
                    self.alive_callback(packet)
                elif packet_type == Packet.TYPE_MESSAGE:
                    self.message_callback(packet)

                # acknowledge the packet was recieved and processed
                self._send_ack(packet)

                # Note to self that this packet has already been processed,
                # and shouldn't be processed again in the future
                self.processed_packets.add(packet_id)

        def run_socket(*args):
            url_base_name = self.server_url.split('https://')[1]
            protocol = "wss"
            if url_base_name in ['localhost', '127.0.0.1']:
                protocol = "ws"
            while self.keep_running:
                try:
                    sock_addr = "{}://{}:{}/".format(
                        protocol, url_base_name, self.port)
                    self.ws = websocket.WebSocketApp(
                        sock_addr,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_disconnect,
                    )
                    self.ws.on_open = on_socket_open
                    self.ws.run_forever(
                        ping_interval=8 * self.HEARTBEAT_RATE
                    )
                    self._ensure_closed()
                except Exception as e:
                    shared_utils.print_and_log(
                        logging.WARN,
                        'Socket error {}, attempting restart'.format(repr(e))
                    )
                time.sleep(0.2)

        # Start listening thread
        self.listen_thread = threading.Thread(
            target=run_socket,
            name='Main-Socket-Recv-Thread'
        )
        self.listen_thread.daemon = True
        self.listen_thread.start()
        time.sleep(1.2)
        start_time = time.time()
        while not self.alive:
            if time.time() - start_time > self.DEF_DEAD_TIME:
                self.server_death_callback()
                raise ConnectionRefusedError(  # noqa F821 we only support py3
                    'Was not able to establish a connection with the server')
            try:
                self._send_world_alive()
            except Exception:
                pass
            time.sleep(self.HEARTBEAT_RATE / 2)

        # Start sending thread
        self.send_thread = threading.Thread(
            target=self.channel_thread,
            name='Main-Socket-Send-Thread'
        )
        self.send_thread.daemon = True
        self.send_thread.start()

    def packet_should_block(self, packet_item):
        """Helper function to determine if a packet is still blocking"""
        t, packet = packet_item
        if time.time() > t:
            return False  # Exceeded blocking time
        if packet.status in [Packet.STATUS_ACK, Packet.STATUS_FAIL]:
            return False  # No longer in blocking status
        return True

    def channel_thread(self):
        """Handler thread for monitoring all channels"""
        # while the thread is still alive
        while not self.is_shutdown:
            if self.ws is None:
                # Waiting for websocket to come back alive
                time.sleep(shared_utils.THREAD_SHORT_SLEEP)
                continue
            for connection_id in self.run.copy():
                if not self.run[connection_id]:
                    continue
                try:
                    # Send a heartbeat if needed
                    self._send_needed_heartbeat(connection_id)
                    # Check if client is still alive
                    if (self.pongs_without_heartbeat[connection_id] >
                            self.missed_pongs):
                        self.run[connection_id] = False
                        # Setup and run the channel sending thread
                        self.threads[connection_id] = threading.Thread(
                            target=self.socket_dead_callback,
                            name='Socket-dead-{}'.format(connection_id),
                            args=self.worker_assign_ids[connection_id]
                        )
                        self.threads[connection_id].daemon = True
                        self.threads[connection_id].start()

                    # Make sure the queue still exists
                    if connection_id not in self.queues:
                        self.run[connection_id] = False
                        break
                    if self.blocking_packets.get(connection_id) is not None:
                        packet_item = self.blocking_packets[connection_id]
                        if not self.packet_should_block(packet_item):
                            self.blocking_packets[connection_id] = None
                        else:
                            continue
                    try:
                        # Get first item in the queue, check if can send it yet
                        item = self.queues[connection_id].get(block=False)
                        t = item[0]
                        if time.time() < t:
                            # Put the item back into the queue,
                            # it's not time to pop yet
                            self._safe_put(connection_id, item)
                        else:
                            # Try to send the packet
                            packet = item[1]
                            if not packet:
                                # This packet was deleted out from under us
                                continue
                            if packet.status is not Packet.STATUS_ACK:
                                # either need to send initial packet
                                # or resend not-acked packet
                                self._send_packet(packet, connection_id, t)
                    except Empty:
                        pass
                except Exception as e:
                    shared_utils.print_and_log(
                        logging.WARN,
                        'Unexpected error occurred in socket handling thread: '
                        '{}'.format(repr(e)),
                        should_print=True,
                    )
            time.sleep(shared_utils.THREAD_SHORT_SLEEP)

    def open_channel(self, worker_id, assignment_id):
        """Opens a channel for a worker on a given assignment, doesn't re-open
        if the channel is already open."""
        connection_id = '{}_{}'.format(worker_id, assignment_id)
        if connection_id in self.queues and self.run[connection_id]:
            shared_utils.print_and_log(
                logging.DEBUG,
                'Channel ({}) already open'.format(connection_id)
            )
            return
        self.run[connection_id] = True
        self.queues[connection_id] = PriorityQueue()
        self.last_sent_heartbeat_time[connection_id] = 0
        self.pongs_without_heartbeat[connection_id] = 0
        self.last_received_heartbeat[connection_id] = None
        self.worker_assign_ids[connection_id] = (worker_id, assignment_id)

    def close_channel(self, connection_id):
        """Closes a channel by connection_id"""
        shared_utils.print_and_log(
            logging.DEBUG,
            'Closing channel {}'.format(connection_id)
        )
        self.run[connection_id] = False
        with self.thread_shutdown_lock:
            if connection_id in self.queues:
                while not self.queues[connection_id].empty():  # Empty queue
                    try:
                        item = self.queues[connection_id].get(block=False)
                        t = item[0]
                        packet = item[1]
                        if not packet:
                            continue
                        packet.requires_ack = False
                        packet.blocking = False
                        if packet.status is not Packet.STATUS_ACK:
                            self._send_packet(packet, connection_id, t)
                    except Empty:
                        break
                # Clean up packets
                with self.packet_map_lock:
                    packet_ids = list(self.packet_map.keys())
                    for packet_id in packet_ids:
                        packet = self.packet_map[packet_id]
                        packet_conn_id = packet.get_receiver_connection_id()
                        if connection_id == packet_conn_id:
                            del self.packet_map[packet_id]
                # Clean up other resources
                del self.queues[connection_id]

    def close_all_channels(self):
        """Closes all channels by clearing the list of channels"""
        shared_utils.print_and_log(logging.DEBUG, 'Closing all channels')
        connection_ids = list(self.queues.keys())
        for connection_id in connection_ids:
            self.close_channel(connection_id)

    def socket_is_open(self, connection_id):
        return connection_id in self.queues

    def queue_packet(self, packet):
        """Queues sending a packet to its intended owner, returns True if
        queued successfully and False if there is no such worker (anymore)
        """
        connection_id = packet.get_receiver_connection_id()
        if not self.socket_is_open(connection_id):
            # Warn if there is no socket to send through for the expected recip
            shared_utils.print_and_log(
                logging.WARN,
                'Can not send packet to worker_id {}: packet queue not found. '
                'Message: {}'.format(connection_id, packet.data)
            )
            return False
        shared_utils.print_and_log(
            logging.DEBUG,
            'Put packet ({}) in queue ({})'.format(packet.id, connection_id)
        )
        # Get the current time to put packet into the priority queue
        with self.packet_map_lock:
            self.packet_map[packet.id] = packet
        item = (time.time(), packet)
        self._safe_put(connection_id, item)
        return True

    def get_status(self, packet_id):
        """Returns the status of a particular packet by id"""
        with self.packet_map_lock:
            if packet_id not in self.packet_map:
                return Packet.STATUS_NONE
            return self.packet_map[packet_id].status

    def _safe_put(self, connection_id, item):
        """Ensures that a queue exists before putting an item into it, logs
        if there's a failure
        """
        if connection_id in self.queues:
            self.queues[connection_id].put(item)
        else:
            item[1].status = Packet.STATUS_FAIL
            shared_utils.print_and_log(
                logging.WARN,
                'Queue {} did not exist to put a message in'.format(
                    connection_id
                )
            )

    def shutdown(self):
        """marks the socket manager as closing, shuts down all channels"""
        self.is_shutdown = True
        self.close_all_channels()
        self.keep_running = False
        self._ensure_closed()


# TODO extract common functionality from the above and make both classes extend
# from the base
class StaticSocketManager(SocketManager):
    """Version of SocketManager that communicates consistently with the world,
    but isn't keeping track of the liveliness of the agents that connect as
    these are single person tasks. Submissions are handled via post rather
    than served over socket, so it doesn't make sense to.
    """
    def channel_thread(self):
        """Handler thread for monitoring all channels to send things to"""
        # while the thread is still alive
        while not self.is_shutdown:
            for connection_id in self.run.copy():
                if not self.run[connection_id]:
                    continue
                try:
                    # Make sure the queue still exists
                    if connection_id not in self.queues:
                        self.run[connection_id] = False
                        break
                    if self.blocking_packets.get(connection_id) is not None:
                        packet_item = self.blocking_packets[connection_id]
                        if not self.packet_should_block(packet_item):
                            self.blocking_packets[connection_id] = None
                        else:
                            continue
                    try:
                        # Get first item in the queue, check if can send it yet
                        item = self.queues[connection_id].get(block=False)
                        t = item[0]
                        if time.time() < t:
                            # Put the item back into the queue,
                            # it's not time to pop yet
                            self._safe_put(connection_id, item)
                        else:
                            # Try to send the packet
                            packet = item[1]
                            if not packet:
                                # This packet was deleted out from under us
                                continue
                            if packet.status is not Packet.STATUS_ACK:
                                # either need to send initial packet
                                # or resend not-acked packet
                                self._send_packet(packet, connection_id, t)
                    except Empty:
                        pass
                except Exception as e:
                    shared_utils.print_and_log(
                        logging.WARN,
                        'Unexpected error occurred in socket handling thread: '
                        '{}'.format(repr(e)),
                        should_print=True,
                    )
            time.sleep(shared_utils.THREAD_SHORT_SLEEP)

    def open_channel(self, worker_id, assignment_id):
        """Opens a channel for a worker on a given assignment, doesn't re-open
        if the channel is already open."""
        connection_id = '{}_{}'.format(worker_id, assignment_id)
        if connection_id in self.queues and self.run[connection_id]:
            shared_utils.print_and_log(
                logging.DEBUG,
                'Channel ({}) already open'.format(connection_id)
            )
            return
        self.run[connection_id] = True
        self.queues[connection_id] = PriorityQueue()
        self.worker_assign_ids[connection_id] = (worker_id, assignment_id)
