/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

/* eslint-disable react/no-direct-mutation-state */

import React from 'react';
import ReactDOM from 'react-dom';

/* ================= Data Model Constants ================= */

// Incoming message commands
const COMMAND_SEND_MESSAGE = 'COMMAND_SEND_MESSAGE';
const COMMAND_SHOW_DONE_BUTTON = 'COMMAND_SHOW_DONE_BUTTON';
const COMMAND_EXPIRE_HIT = 'COMMAND_EXPIRE_HIT';
const COMMAND_SUBMIT_HIT = 'COMMAND_SUBMIT_HIT';
const COMMAND_CHANGE_CONVERSATION = 'COMMAND_CHANGE_CONVERSATION';
const COMMAND_RESTORE_STATE = 'COMMAND_RESTORE_STATE';
const COMMAND_INACTIVE_HIT = 'COMMAND_INACTIVE_HIT';
const COMMAND_INACTIVE_DONE = 'COMMAND_INACTIVE_DONE';

// Socket function names
const SOCKET_OPEN_STRING = 'socket_open'; // fires when a socket opens
const SOCKET_DISCONNECT_STRING = 'disconnect'; // fires if a socket disconnects
const SOCKET_NEW_PACKET_STRING = 'new packet'; // fires when packets arrive
const SOCKET_ROUTE_PACKET_STRING = 'route packet'; // to send outgoing packets
const SOCKET_AGENT_ALIVE_STRING = 'agent alive'; // to send alive packets

// Message types
const MESSAGE_TYPE_MESSAGE = 'MESSAGE';
const MESSAGE_TYPE_COMMAND = 'COMMAND';

// Packet types
const TYPE_ACK = 'ack';
const TYPE_MESSAGE = 'message';
const TYPE_HEARTBEAT = 'heartbeat';
const TYPE_PONG = 'pong'; // For messages back from the router on heartbeat
const TYPE_ALIVE = 'alive';

/* ================= Local Constants ================= */

const SEND_THREAD_REFRESH = 100;
const ACK_WAIT_TIME = 2000; // Check for acknowledge every 2 seconds
const STATUS_ACK = 'ack';
const STATUS_INIT = 'init';
const STATUS_SENT = 'sent';
const CONNECTION_DEAD_MISSING_PONGS = 25;
const REFRESH_SOCKET_MISSING_PONGS = 10;
const HEARTBEAT_TIME = 4000; // One heartbeat every 4 seconds

/* ============== Priority Queue Data Structure ============== */

// Initializing a 'priority queue'
function PriorityQueue() {
  this.data = [];
}

// Pushes an element as far back as it needs to go in order to insert
PriorityQueue.prototype.push = function(element, priority) {
  priority = +priority;
  for (var i = 0; i < this.data.length && this.data[i][0] < priority; i++);
  this.data.splice(i, 0, [element, priority]);
};

// Remove and return the front element of the queue
PriorityQueue.prototype.pop = function() {
  return this.data.shift();
};

// Show the front element of the queue
PriorityQueue.prototype.peek = function() {
  return this.data[0];
};

// gets the size of the queue
PriorityQueue.prototype.size = function() {
  return this.data.length;
};

/* ================= Utility functions ================= */

// log the data when the verbosity is greater than the level
// (where low level is high importance)
// Levels: 0 - Error, unusual behavior, something worth notifying
//         1 - Server interactions on the message level, commands
//         2 - Server interactions on the heartbeat level
//         3 - Potentially important function calls
//         4 - Practically anything
function log(data, level) {
  if (verbosity >= level) {
    console.log(data);
  }
}

// Checks to see if given conversation_id is for a waiting world
function isWaiting(conversation_id) {
  return conversation_id != null && conversation_id.indexOf('w_') !== -1;
}

// Checks to see if given conversation_id is for a task world
function isTask(conversation_id) {
  return conversation_id != null && conversation_id.indexOf('t_') !== -1;
}

// Checks to see if given conversation_id is for an onboarding world
function isOnboarding(conversation_id) {
  return conversation_id != null && conversation_id.indexOf('o_') !== -1;
}

// Generate a random id
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = (Math.random() * 16) | 0,
      v = c == 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

class SocketHandler extends React.Component {
  constructor(props) {
    super(props);
    this.q = new PriorityQueue();
    this.socket = null;
    this.state = {
      heartbeat_id: null, // Timeout id for heartbeat thread
      socket_closed: false, // Flag for intentional socket closure
      setting_socket: false, // Flag for socket setup being underway
      pongs_without_heartbeat: 0, // Pongs from router w/o hb from MTurkManager
      heartbeats_without_pong: 0, // HBs to the router without a pong back
      packet_map: {}, // Map from packet ids to packets
      packet_callback: {}, // Map from packet ids to callbacks
      blocking_id: null, // Packet id of a blocking message underway
      blocking_sent_time: null, // Time blocking message was sent
      blocking_intend_send_time: null, // Time of blocking message priority
      displayed_messages: [], // Message ids that are already displayed
      message_request_time: null, // Last request for a message to find delay
    };
  }

  /*************************************************
   * ----- Packet sending/management functions ------
   *
   * The following functions comprise the outgoing
   * packet management system. Messages are enqueued
   * using sendPacket. The sendingThread works through
   * the queued packets and passes them to sendHelper
   * when appropriate (unblocked, not already sent).
   * sendHelper handles updating blocking state during
   * a send, and then pushes the packet through using
   * safePacketSend.
   **/

  // Push a packet through the socket
  safePacketSend(packet) {
    if (this.socket.readyState == 0) {
      return;
    } else if (this.socket.readyState > 1) {
      log('Socket not in ready state, restarting if possible', 2);
      try {
        this.socket.close();
      } catch (e) {
        /* Socket already terminated */
      }
      this.setupSocket();
      return;
    }
    try {
      this.socket.send(JSON.stringify(packet));
    } catch (e) {
      log('Had error ' + e + ' sending message, trying to restart', 2);
      try {
        this.socket.close();
      } catch (e) {
        /* Socket already terminated */
      }
      this.setupSocket();
    }
  }

  // Wrapper around packet sends that handles managing blocking and state
  // updates, as well as not sending packets that have already been sent
  sendHelper(msg, queue_time) {
    // Don't act on acknowledged packets
    if (msg.status !== STATUS_ACK) {
      // Find the event to send to
      let event_name = SOCKET_ROUTE_PACKET_STRING;
      if (msg.type === TYPE_ALIVE) {
        event_name = SOCKET_AGENT_ALIVE_STRING;
      }
      this.safePacketSend({ type: event_name, content: msg });

      if (msg.require_ack) {
        if (msg.blocking) {
          // Block the thread
          this.setState({
            blocking_id: msg.id,
            blocking_sent_time: Date.now(),
            blocking_intend_send_time: queue_time,
          });
        } else {
          // Check to see if the packet is acknowledged in the future
          this.q.push(msg, queue_time + ACK_WAIT_TIME);
        }
      }
    }
  }

  // Thread checks the message queue and handles pushing out new messages
  // as they are added to the queue
  sendingThread() {
    if (this.state.socket_closed) {
      return;
    }
    let blocking_id = this.state.blocking_id;

    // Can't act if something is currently blocking
    if (blocking_id === null) {
      // Can't act on an empty queue
      if (this.q.size() > 0) {
        // Can't act if the send time for the next thing to send
        // is in the future
        if (Date.now() > this.q.peek()[1]) {
          var item = this.q.pop();
          var msg = item[0];
          var t = item[1];
          this.sendHelper(msg, t);
        }
      }
    } else {
      let blocking_sent_time = this.state.blocking_sent_time;
      let packet_map = this.state.packet_map;
      let blocking_intend_send_time = this.state.blocking_intend_send_time;
      // blocking on packet `blocking_id`
      // See if we've waited too long for the packet to be acknowledged
      if (Date.now() - blocking_sent_time > ACK_WAIT_TIME) {
        log('Timeout: ' + blocking_id, 1);
        // Send the packet again now
        var m = packet_map[blocking_id];
        if (m.status != STATUS_ACK) {
          // Ensure that the send retries, we wouldn't be here if the ACK worked
          m.status = STATUS_SENT;
          this.sendHelper(packet_map[blocking_id], blocking_intend_send_time);
        }
      }
    }
  }

  // Enqueues a message for sending, registers the message and callback
  sendPacket(type, data, require_ack, blocking, callback) {
    var time = Date.now();
    var id = uuidv4();

    var msg = {
      id: id,
      type: type,
      sender_id: this.props.worker_id,
      assignment_id: this.props.assignment_id,
      conversation_id: this.props.conversation_id,
      receiver_id: '[World_' + TASK_GROUP_ID + ']',
      data: data,
      status: STATUS_INIT,
      require_ack: require_ack,
      blocking: blocking,
    };

    this.q.push(msg, time);
    this.state.packet_map[id] = msg;
    this.state.packet_callback[id] = callback;
  }

  // Required function - The BaseApp class will call this function to enqueue
  // packet sends that are requested by the frontend user (worker)
  handleQueueMessage(text, task_data, callback, is_system = false) {
    let new_message_id = uuidv4();
    let duration = null;
    if (!is_system && this.state.message_request_time != null) {
      let cur_time = new Date().getTime();
      duration = cur_time - this.state.message_request_time;
      this.setState({ message_request_time: null });
    }
    this.sendPacket(
      TYPE_MESSAGE,
      {
        text: text,
        task_data: task_data,
        id: this.props.agent_id,
        message_id: new_message_id,
        episode_done: false,
        duration: duration,
      },
      true,
      true,
      msg => {
        if (!is_system) {
          this.props.messages.push(msg.data);
          this.props.onSuccessfulSend();
        }
        if (callback !== undefined) {
          callback();
        }
      }
    );
  }

  // way to send alive packets when expected to
  sendAlive() {
    this.sendPacket(
      TYPE_ALIVE,
      {
        hit_id: this.props.hit_id,
        assignment_id: this.props.assignment_id,
        worker_id: this.props.worker_id,
        conversation_id: this.props.conversation_id,
      },
      true,
      true,
      () => {
        this.props.onConfirmInit();
        this.props.onStatusChange('connected');
      }
    );
  }

  /**************************************************
   * ----- Packet reception infra and functions ------
   *
   * The following functions are all related to
   * handling incoming messages. parseSocketMessage
   * filters out actions based on the type of message,
   * including updating the state of health when
   * recieving pongs and heartbeats. handleNewMessage
   * is used to process an incoming Message that is
   * supposed to be entered into the chat. handleCommand
   * handles the various commands that are sent from
   * the ParlAI host to change the frontend state.
   **/

  parseSocketMessage(event) {
    let msg = JSON.parse(event.data)['content'];
    if (msg.type === TYPE_HEARTBEAT) {
      // Heartbeats ensure we're not disconnected from the server
      log('received heartbeat: ' + msg.id, 5);
      this.setState({ pongs_without_heartbeat: 0 });
    } else if (msg.type === TYPE_PONG) {
      // Messages incoming from the router, ensuring that we're at least
      // connected to it.
      this.setState({
        pongs_without_heartbeat: this.state.pongs_without_heartbeat + 1,
        heartbeats_without_pong: 0,
      });
      if (this.state.pongs_without_heartbeat >= 3) {
        this.props.onStatusChange('reconnecting_server');
      }
    } else if (msg.type === TYPE_ACK) {
      // Acks update messages so that we don't resend them
      log('received ack: ' + msg.id, 3);
      if (msg.id === this.state.blocking_id) {
        // execute ack callback if it exists
        if (this.state.packet_callback[msg.id]) {
          this.state.packet_callback[msg.id](this.state.packet_map[msg.id]);
        }
        this.setState({ blocking_id: null, blocking_sent_time: null });
      }
      this.state.packet_map[msg.id].status = STATUS_ACK;
    } else if (msg.type === TYPE_MESSAGE) {
      // Acknowledge the message, then act on it
      this.safePacketSend({
        type: SOCKET_ROUTE_PACKET_STRING,
        content: {
          id: msg.id,
          sender_id: msg.receiver_id,
          receiver_id: msg.sender_id,
          assignment_id: this.props.assignment_id,
          conversation_id: this.props.conversation_id,
          type: TYPE_ACK,
          data: null,
        },
      });
      log(msg, 3);
      if (msg.data.type === MESSAGE_TYPE_COMMAND) {
        this.handleCommand(msg.data);
      } else if (msg.data.type === MESSAGE_TYPE_MESSAGE) {
        this.handleNewMessage(msg.id, msg.data);
      }
    }
  }

  // Handles an incoming message
  handleNewMessage(new_message_id, message) {
    if (message.text === undefined) {
      message.text = '';
    }
    var agent_id = message.id;
    var message_text = message.text;
    if (this.state.displayed_messages.indexOf(new_message_id) !== -1) {
      // This message has already been seen and put up into the chat
      log(new_message_id + ' was a repeat message', 1);
      return;
    }

    log('New message, ' + new_message_id + ' from agent ' + agent_id, 1);
    this.state.displayed_messages.push(new_message_id);
    this.props.onNewMessage(message);
    if (message.task_data !== undefined) {
      let has_context = false;
      if (!this.props.run_static) {
        // We expect static tasks to recieve task_data and display in
        // the main pane, but dynamic tasks have other options.
        // TODO extract above if block out after separating static and dynamic
        // Socket managers
        for (let key of Object.keys(message.task_data)) {
          if (key !== 'respond_with_form') {
            has_context = true;
          }
        }
      }

      message.task_data.last_update = new Date().getTime();
      message.task_data.has_context = has_context;
      this.props.onNewTaskData(message.task_data);
    }
    this.setState({ displayed_messages: this.state.displayed_messages });
  }

  // Handle incoming command messages
  handleCommand(msg) {
    let command = msg['text'];
    log('Received command ' + command, 1);
    if (command === COMMAND_SEND_MESSAGE) {
      // Update UI to wait for the worker to submit a message
      this.props.onRequestMessage();
      if (this.state.message_request_time === null) {
        this.props.playNotifSound();
      }
      this.setState({ message_request_time: new Date().getTime() });
      log('Waiting for worker input', 4);
    } else if (command === COMMAND_SHOW_DONE_BUTTON) {
      // Update the UI to show the done button
      this.props.onTaskDone();
    } else if (command === COMMAND_INACTIVE_DONE) {
      // Update the UI to show done with additional inactive text
      this.closeSocket();
      // Call correct UI renderers
      this.props.onInactiveDone(msg['inactive_text']);
    } else if (command === COMMAND_SUBMIT_HIT) {
      // Force the hit to submit as done
      this.props.onForceDone();
    } else if (command === COMMAND_EXPIRE_HIT) {
      // Expire the hit unless it has already been marked as done
      if (!this.props.task_done) {
        this.props.onExpire(msg['inactive_text']);
        this.closeSocket();
      }
    } else if (command === COMMAND_INACTIVE_HIT) {
      // Disable the hit, show the correct message
      this.props.onExpire(msg['inactive_text']);
      this.closeSocket();
    } else if (command === COMMAND_RESTORE_STATE) {
      // Restore the messages from inside the data, call the command if needed
      let messages = msg['messages'];
      for (var i = 0; i < messages.length; i++) {
        this.handleNewMessage(messages[i]['message_id'], messages[i]);
      }

      let last_command = msg['last_command'];
      if (last_command) {
        this.handleCommand(last_command);
      }
    } else if (command === COMMAND_CHANGE_CONVERSATION) {
      // change the conversation, refresh if needed
      let conversation_id = msg['conversation_id'];
      log('new conversation_id: ' + conversation_id, 3);
      let agent_id = msg['agent_id'];
      let world_state = null;
      if (isWaiting(conversation_id)) {
        world_state = 'waiting';
      } else if (isOnboarding(conversation_id)) {
        world_state = 'onboarding';
      } else if (isTask(conversation_id)) {
        world_state = 'task';
      }
      this.props.onConversationChange(world_state, conversation_id, agent_id);
      this.sendAlive();
    }
  }

  /**************************************************
   * ---------- socket lifecycle functions -----------
   *
   * These functions comprise the socket's ability to
   * start up, check it's health, restart, and close.
   * setupSocket registers the socket handlers, and
   * when the socket opens it sets a timeout for
   * having the ParlAI host ack the alive (failInitialize).
   * It also starts heartbeats (using heartbeatThread) if
   * they aren't already underway. On an error the socket
   * restarts. The heartbeat thread manages sending
   * heartbeats and tracking when the router responds
   * but the parlai host does not.
   **/

  // Sets up and registers the socket and the callbacks
  setupSocket() {
    if (this.state.setting_socket || this.state.socket_closed) {
      return;
    }

    // Note we are setting up the socket, ready to clear on failure
    this.setState({ setting_socket: true });
    window.setTimeout(() => this.setState({ setting_socket: false }), 4000);

    let url = window.location;
    if (url.hostname == 'localhost') {
      // Localhost can't always handle secure websockets, so we special case
      this.socket = new WebSocket('ws://' + url.hostname + ':' + url.port);
    } else {
      this.socket = new WebSocket('wss://' + url.hostname + ':' + url.port);
    }
    // TODO if socket setup fails here, see if 404 or timeout, then check
    // other reasonable domain. If that succeeds, assume the heroku server
    // died.

    this.socket.onmessage = event => {
      this.parseSocketMessage(event);
    };

    this.socket.onopen = () => {
      log('Server connected.', 2);
      let setting_socket = false;
      window.setTimeout(() => this.sendAlive(), 100);
      window.setTimeout(() => this.failInitialize(), 10000);
      let heartbeat_id = null;
      if (this.state.heartbeat_id == null) {
        heartbeat_id = window.setInterval(
          () => this.heartbeatThread(),
          HEARTBEAT_TIME
        );
      } else {
        heartbeat_id = this.state.heartbeat_id;
      }
      this.setState({
        heartbeat_id: heartbeat_id,
        setting_socket: setting_socket,
      });
    };

    this.socket.onerror = () => {
      log('Server disconnected.', 3);
      try {
        this.socket.close();
      } catch (e) {
        log('Server had error ' + e + ' when closing after an error', 1);
      }
      window.setTimeout(() => this.setupSocket(), 500);
    };

    this.socket.onclose = () => {
      log('Server closing.', 3);
      this.props.onStatusChange('disconnected');
    };
  }

  failInitialize() {
    if (this.props.initialization_status == 'initializing') {
      this.props.onFailInit();
    }
  }

  closeSocket() {
    if (!this.state.socket_closed) {
      log('Socket closing', 3);
      this.socket.close();
      this.setState({ socket_closed: true });
    } else {
      log('Socket already closed', 2);
    }
  }

  componentDidMount() {
    this.setupSocket();
    window.setInterval(() => this.sendingThread(), SEND_THREAD_REFRESH);
  }

  // Thread sends heartbeats through the socket for as long we are connected
  heartbeatThread() {
    // TODO fail properly and update state to closed when the host dies for
    // a long enough period. Once this says "disconnected" it should be
    // disconnected
    if (this.state.socket_closed || this.props.run_static) {
      // No reason to keep a heartbeat if the socket is closed
      window.clearInterval(this.state.heartbeat_id);
      this.setState({ heartbeat_id: null });
      return;
    }

    let pongs_without_heartbeat = this.state.pongs_without_heartbeat;
    if (pongs_without_heartbeat == REFRESH_SOCKET_MISSING_PONGS) {
      pongs_without_heartbeat += 1;
      try {
        this.socket.close(); // Force a socket close to make it reconnect
      } catch (e) {
        /* Socket already terminated */
      }
      window.clearInterval(this.state.heartbeat_id);
      this.setState({
        pongs_without_heartbeat: pongs_without_heartbeat,
        heartbeat_id: null,
      });
      this.setupSocket();
    }

    // Check to see if we've disconnected from the server
    if (this.state.pongs_without_heartbeat > CONNECTION_DEAD_MISSING_PONGS) {
      this.closeSocket();
      let done_text =
        "Our server appears to have gone down during the \
        duration of this HIT. Please send us a message if you've done \
        substantial work and we can find out if the hit is complete enough to \
        compensate.";
      window.clearInterval(this.state.heartbeat_id);
      this.setState({ heartbeat_id: null });
      this.props.onExpire(done_text);
    }

    var hb = {
      id: uuidv4(),
      receiver_id: '[World_' + TASK_GROUP_ID + ']',
      assignment_id: this.props.assignment_id,
      sender_id: this.props.worker_id,
      conversation_id: this.props.conversation_id,
      type: TYPE_HEARTBEAT,
      data: null,
    };

    this.safePacketSend({ type: SOCKET_ROUTE_PACKET_STRING, content: hb });

    this.setState({
      heartbeats_without_pong: this.state.heartbeats_without_pong + 1,
    });
    if (this.state.heartbeats_without_pong >= 12) {
      this.closeSocket();
    } else if (this.state.heartbeats_without_pong >= 3) {
      this.props.onStatusChange('reconnecting_router');
    }
  }

  // No rendering component for the SocketHandler
  render() {
    return null;
  }
}

export default SocketHandler;
