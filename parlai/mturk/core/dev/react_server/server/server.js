/* Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
'use strict';

const bodyParser = require('body-parser');
const express = require('express');
const fs = require('fs');
const http = require('http');
const nunjucks = require('nunjucks');
var request = require('request');
const WebSocket = require('ws');

const task_directory_name = 'static';

const PORT = process.env.PORT || 3000;

import { get_message_for_status } from './constants/status_messages.js'
import * as agent_states from './constants/agent_states.js'

// Initialize app
const app = express();
app.use(bodyParser.text());
app.use(
  bodyParser.urlencoded({
    extended: true,
  })
);
app.use(bodyParser.json());

nunjucks.configure(task_directory_name, {
  autoescape: true,
  express: app,
});

// Generate a random id
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = (Math.random() * 16) | 0,
      v = c == 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}


// ===================== <Agent state> ====================

const AGENT_TIMEOUT_TIME = 20000; // 20 seconds

function is_final_status(status) {
  return [
    agent_states.STATUS_DONE, agent_states.STATUS_DISCONNECT,
    agent_states.STATUS_PARTNER_DISCONNECT, agent_states.STATUS_EXPIRED,
    agent_states.STATUS_RETURNED, agent_states.STATUS_PARLAI_DISCONNECT,
  ].includes(status);
}

// This is used to track the local state of an MTurk agent as determined by
// the python backend. It contains the current conversation id,
// the current task status, a mapping from conversation id to
// incoming messages, another to message history, and a repeat message set.
//
// Utility functions exist to process an incoming message (which puts it
// into the correct data structures after ensuring it is new), and to update
// the agent status (as this has impact on the conversation_id, and memory
// cleanup)
class LocalAgentState {
  constructor(connection_id) {
    this.status = agent_states.STATUS_NONE;
    this.agent_id = null;
    this.conversation_id = null;
    this.unsent_messages = {};
    this.previous_messages = {};
    this.received_message_ids = new Set([]);
    this.connection_id = connection_id;
    this.last_heartbeat = Date.now();
    this.done_text = null;
  }

  _init_conversation(conversation_id) {
    this.previous_messages[conversation_id] =
      this.previous_messages[conversation_id] || [];
    this.unsent_messages[conversation_id] =
      this.unsent_messages[conversation_id] || [];
  }

  _cleanup_state() {
    // When a task is 'done', we can clean up the state.
    this.unsent_messages = {};
    this.previous_messages = {};
    this.received_message_ids = new Set([]);

    // Remove self from active connections
    if (active_connections.has(this.connection_id)) {
      active_connections.delete(this.connection_id);
    }
  }

  get_reconnect_packet() {
    let done_text = this.done_text;
    if (!done_text && is_final_status(this.status)) {
      done_text = get_message_for_status(this.status);
    }

    return {
      conversation_id: this.conversation_id,
      messages: this.get_previous_messages_for(this.conversation_id),
      agent_status: this.status,
      done_text: done_text,
      agent_id: this.agent_id,
    }
  }

  get_previous_messages_for(conversation_id) {
    if (!conversation_id) {
      return [];
    }

    if (!this.previous_messages[conversation_id]) {
      return [];
    }

    return this.previous_messages[conversation_id];
  }

  get_update_from_state_change(state_change) {
    // Updates this state based on the state change, and provides an
    // update packet to send forward to the client as well
    let new_conversation_id =
      state_change['conversation_id'] || this.conversation_id;
    let new_status = state_change['agent_status'] || this.status;
    let done_text = state_change['done_text'] || this.done_text;
    let agent_id = state_change['agent_id'] || this.agent_id;

    let update_packet = {};
    let needs_update = false;
    if (new_conversation_id != this.conversation_id) {
      needs_update = true;
      update_packet['conversation_id'] = new_conversation_id;
      this.conversation_id = new_conversation_id;
      this._init_conversation(new_conversation_id);
    }

    if (new_status != this.status) {
      needs_update = true;
      update_packet['agent_status'] = new_status;
      this.status = new_status;

      if (is_final_status(new_status)) {
        update_packet['final'] = true;
        this._cleanup_state();
      }

      if (new_status == agent_states.STATUS_STATIC) {
        // Status static needs special cleanup to keep state but stop tracking
        let sendable_messages = this.get_sendable_messages()
        if (sendable_messages.length > 0) {
          let packet = {'messages': sendable_messages};
          _send_message(this.connection_id, MESSAGE_BATCH, packet);
        }
        if (active_connections.has(this.connection_id)) {
          active_connections.delete(this.connection_id);
        }
      }
    }

    if (done_text != this.done_text) {
      needs_update = true;
      update_packet['done_text'] = done_text;
      this.done_text = done_text;
    }

    if (agent_id != this.agent_id) {
      needs_update = true;
      update_packet['agent_id'] = agent_id;
      this.agent_id = agent_id;
    }

    if (needs_update) {
      return update_packet;
    } else {
      return null;
    }
  }

  new_message(msg) {
    if (this.received_message_ids.has(msg['id'])) {
      return;  // This message has already been added
    }

    if (is_final_status(this.status)) {
      // There's no message queue to deliver this message to anymore
      console.log("msg queue was already cleaned up");
      console.log(msg);
      return;
    }

    // rare new_message called before set_status race condition
    if (this.previous_messages[msg['conversation_id']] === undefined) {
      this._init_conversation(msg['conversation_id']);
    }

    this.unsent_messages[msg['conversation_id']].push(msg);
    this.previous_messages[msg['conversation_id']].push(msg);
  }

  get_sendable_messages() {
    if (this.conversation_id == null) {
      return [];
    }
    let unsent_messages = this.unsent_messages[this.conversation_id] || [];
    let return_messages = [];
    while (unsent_messages.length > 0) {
      return_messages.push(unsent_messages.shift())
    }
    return return_messages;
  }

  add_sent_message(sent_message) {
    let conversation_id = sent_message['conversation_id']
    if (this.previous_messages[conversation_id] === undefined) {
      this._init_conversation(conversation_id)
    }
    this.previous_messages[conversation_id].push(sent_message);
  }

  get_update_from_heartbeat(heartbeat) {
    // Handle creating an update packet for a given heartbeat if data doesn't
    // align. Returns null if everything is in order.
    var client_message_count = heartbeat['received_message_count'];
    var client_status = heartbeat['agent_status']
    var client_conversation_id = heartbeat['conversation_id']
    var client_done_text = heartbeat['done_text']

    if (client_conversation_id != this.conversation_id) {
      // Make a new packet with all the missing information
      return {
        conversation_id: this.conversation_id,
        messages: this.get_previous_messages_for(this.conversation_id),
        agent_status: this.status,
        done_text: this.done_text,
        agent_id: this.agent_id,
      }
    }

    // Updating remaining fields if needed
    let update_packet = {last_parlai_ping: world_last_ping};
    if (this.conversation_id !== null) {
      let previous_messages =
        this.get_previous_messages_for(this.conversation_id);
      if (client_message_count < previous_messages.length) {
        let missing_messages = previous_messages.slice(client_message_count);
        update_packet['messages'] = missing_messages;
      }
    }

    if (client_status != this.status) {
      update_packet['agent_status'] = this.status;
    }

    if (client_done_text != this.done_text) {
      update_packet['done_text'] = this.done_text;
    }

    return update_packet;
  }

  mark_heartbeat() {
    this.last_heartbeat = Date.now();
  }
}

// ======================= <Socket> =======================

// Socket function types
// FIXME move to constants
const AGENT_MESSAGE = 'agent message'  // Message from an agent
const WORLD_MESSAGE = 'world message'  // Message from world to agent
const HEARTBEAT = 'heartbeat'   // Heartbeat from agent, carries current state
const WORLD_PING = 'world ping'  // Ping from the world for this server uptime
const SERVER_PONG = 'server pong'  // pong to confirm uptime
const MESSAGE_BATCH = 'message batch' // packet containing batch of messages
const AGENT_DISCONNECT = 'agent disconnect'  // Notes an agent disconnecting
const SNS_MESSAGE = 'sns message'   // packet from an SNS message
const SUBMIT_MESSAGE = 'submit message'  // packet from done POST
const AGENT_STATE_CHANGE = 'agent state change'  // state change from parlai
const AGENT_ALIVE = 'agent alive'  // packet from an agent alive event
const UPDATE_STATE = 'update state'  // packet for updating agent client state

// The state gets passed forward to the server whenever anyone connects
// Messages and a history are kept until an agent hits a final state
//
// Incoming messages get organized into the intended conversation id
// for an agent. This means we can restore the state depending on the
// current conversation.
//
// Only backwards messages are initial alives, message sends,
// and disconnects.

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Track connections
var connection_id_to_socket = {};
var socket_id_to_connection_id = {};
var world_message_queue = [];
var world_last_ping = 0;
var main_thread_timeout = null;

// Used to store the id to refer to the ParlAI socket with, the world_id is
// initialized by the first socket connection from ParlAI to the router server.
var world_id = null;

// This is a mapping of connection id -> state for an MTurk agent
var connection_id_to_agent_state = {};


var NOTIF_ID = 'MTURK_NOTIFICATIONS';

// Handles sending a message through the socket
// FIXME create a version that handles wrapping into a packet?
function _send_message(connection_id, event_name, event_data) {
  // Find the connection's socket
  var socket = connection_id_to_socket[connection_id];
  // Server does not have information about this worker. Should wait for this
  // worker's agent_alive event instead.
  if (!socket) {
    return;
  }

  if (socket.readyState == 3) {
    return;
  }

  if (event_data.receiver_id === undefined) {
    event_data.receiver_id = connection_id;
  }

  var packet = {
    type: event_name,
    content: event_data,
  };
  // Send the message through
  socket.send(JSON.stringify(packet), function ack(error) {
    if (error === undefined) {
      return;
    }
    setTimeout(function() {
      socket.send(JSON.stringify(packet), function ack2(error2) {
        if (error2 === undefined) {
          return;
        }
      });
    }, 500);
  });
}

// Connection ids differ when they are heading to or from the world, these
// functions let the rest of message sending logic remain consistent
function _get_to_conn_id(data) {
  var receiver_id = data['receiver_id'];
  if (receiver_id && receiver_id.startsWith('[World')) {
    return receiver_id;
  } else {
    return receiver_id + '_' + data['assignment_id'];
  }
}

function _get_from_conn_id(data) {
  var sender_id = data['sender_id'];
  if (sender_id && sender_id.startsWith('[World')) {
    return sender_id;
  } else {
    return sender_id + '_' + data['assignment_id'];
  }
}

function handle_pong(data) {
  // Return a pong which is used to ensure that the heroku server is still live
  data['type'] = 'pong';
  data['id'] = uuidv4();
  var out_connection_id = _get_from_conn_id(data);

  world_last_ping = Date.now();
  _send_message(out_connection_id, SERVER_PONG, data);
}

// Agent alive events are handled by registering the agent to a connection_id
// and then forwarding the alive to the world if it came from a client. If
// there is already an agent state for this agent, we handle the reconnection
// event ourselves.
function handle_alive(socket, data) {
  var sender_id = data['sender_id'];
  var in_connection_id = _get_from_conn_id(data);
  connection_id_to_socket[in_connection_id] = socket;
  socket_id_to_connection_id[socket.id] = in_connection_id;

  if (!(sender_id && sender_id.startsWith('[World'))) {
    // Worker has connected, check to see if the connection exists
    let agent_state = connection_id_to_agent_state[in_connection_id];
    if (agent_state === undefined) {
      // Worker is connecting for the first time, init state and forward alive
      _send_message(world_id, AGENT_ALIVE, data);
      let new_state = new LocalAgentState(in_connection_id);
      connection_id_to_agent_state[in_connection_id] = new_state;
      active_connections.add(in_connection_id)
    } else if (agent_state.conversation_id != data['conversation_id']) {
      // Agent is reconnecting, and needs to be updated in full
      let update_packet = agent_state.get_reconnect_packet();
      _send_message(in_connection_id, UPDATE_STATE, update_packet);
    }
  } else {
    world_id = in_connection_id;
    // Send alive packets to the world, but not from the world
    socket.send(
      JSON.stringify({ type: 'conn_success', content: 'Socket is open!' })
    );
    if (main_thread_timeout === null) {
      main_thread_timeout = setTimeout(main_thread, 50);
    }
  }
}

function handle_agent_heartbeat(hb) {
  // Check heartbeat data against the expected state, create update packets if
  // something is wrong.
  var in_connection_id = _get_from_conn_id(hb);
  var agent_state = connection_id_to_agent_state[in_connection_id];
  if (agent_state === undefined) {
    // Hasn't alived yet, just return, no state to handle
    return;
  }
  var update_packet = agent_state.get_update_from_heartbeat(hb);
  _send_message(in_connection_id, UPDATE_STATE, update_packet);

  agent_state.mark_heartbeat();
}

function handle_batch_to_agents(incoming_messages) {
  // Process individual messages and add them to the correct agent
  // queues. Doesn't actually forward the messages
  // TODO update when messages from SocketManager actually come in batches
  incoming_messages = [incoming_messages];
  for (let i in incoming_messages) {
    let message = incoming_messages[i];
    var out_connection_id = _get_to_conn_id(message);
    let agent_state = connection_id_to_agent_state[out_connection_id];
    if (agent_state === undefined) {
      // sending message to an agent that doesn't exist?
      console.log("msg was sent to non-existent" + out_connection_id);
      console.log(message);
    } else {
      agent_state.new_message(message);
    }
  }
}

function handle_batch_to_world(agent_message) {
  // Process single message and add it to the batch for the world while
  // updating the local state of an agent to have the new message
  let in_connection_id = _get_from_conn_id(agent_message);
  let agent_state = connection_id_to_agent_state[in_connection_id];
  if (agent_state === undefined) {
    // sending message to an agent that doesn't exist?
    console.log("msg was recieved from non-existent" + in_connection_id);
    console.log(agent_message);
  } else {
    agent_state.add_sent_message(agent_message);
  }

  // Simply adds message to the end, will be shifted from front
  world_message_queue.push(agent_message);
}

function handle_agent_state_update(state_change_packet) {
  // Process this event properly, generate any required update commands
  // for the frontend agent. Generally this should just involve sending a
  // json of the given state for the current conversation.
  let out_connection_id = _get_to_conn_id(state_change_packet);
  let agent_state = connection_id_to_agent_state[out_connection_id];
  if (agent_state === undefined) {
    // sending state update to an agent that doesn't exist?
    console.log("msg was sent to non-existent" + out_connection_id);
    console.log(state_change_packet);
  } else if (!is_final_status(agent_state.status)) {
    let state_change = state_change_packet.data;
    let update_packet = agent_state.get_update_from_state_change(state_change);
    // Only state updates to agents that are still alive
    if (active_connections.has(out_connection_id)) {
      _send_message(out_connection_id, UPDATE_STATE, update_packet);
    }
  }
}

// Register handlers
wss.on('connection', function(socket) {
  console.log('Client connected');
  // Disconnects are logged
  socket.on('disconnect', function() {
    var connection_id = socket_id_to_connection_id[socket.id];
    console.log('Client disconnected: ' + connection_id);
  });

  socket.on('error', err => {
    console.log('Caught socket error');
    console.log(err);
  });

  // handles routing a packet to the desired recipient
  socket.on('message', function(data) {
    try {
      // FIXME It's somewhat annoying that these constants come up and
      // redefined all over the place, would be useful to have a singular
      // source for the API
      data = JSON.parse(data);
      if (data['type'] == AGENT_ALIVE) {
        handle_alive(socket, data['content']);
      } else if (data['type'] == AGENT_MESSAGE) {
        handle_batch_to_world(data['content']);
      } else if (data['type'] == AGENT_STATE_CHANGE) {
        handle_agent_state_update(data['content']);
      } else if (data['type'] == WORLD_MESSAGE) {
        handle_batch_to_agents(data['content']);
      } else if (data['type'] == HEARTBEAT) {
        handle_agent_heartbeat(data['content']);
      } else if (data['type'] == WORLD_PING) {
        handle_pong(data['content']);
      }
    } catch (error) {
      console.log('Transient error on message');
      console.log(error);
    }
  });
});

server.listen(PORT, function() {
  console.log('Listening on %d', server.address().port);
});

// ======================= </Socket> =======================

// ======================= <Threads> =======================

var active_connections = new Set([]);

// TODO add crash checking around this thread?
function main_thread() {
  // Handle active connections message sends
  for (const connection_id of active_connections) {
    let agent_state = connection_id_to_agent_state[connection_id];
    let sendable_messages = agent_state.get_sendable_messages();
    if (sendable_messages.length > 0) {
      let packet = {'messages': sendable_messages};
      _send_message(connection_id, MESSAGE_BATCH, packet);
    }
  }

  // Handle sending batches to the world
  let world_messages = []
  while (world_message_queue.length > 0) {
    world_messages.push(world_message_queue.shift());
  }
  if (world_messages.length > 0) {
    let msg = {
      id: uuidv4(),
      type: 'message',
      sender_id: null,
      assignment_id: null,
      conversation_id: 'MessageBatch',
      receiver_id: world_id,
      data: {'messages': world_messages},
    };
    _send_message(world_id, MESSAGE_BATCH, msg);
  }

  // Handle submitting disconnect events
  for (const connection_id of active_connections) {
    let agent_state = connection_id_to_agent_state[connection_id];
    // Non-static tasks should keep tabs on the sockets
    if (agent_state.status != agent_states.STATUS_STATIC) {
      let now = Date.now();
      if (now - agent_state.last_heartbeat > AGENT_TIMEOUT_TIME) {
        let msg = {
          id: uuidv4(),
          type: 'message',
          sender_id: null,
          assignment_id: null,
          conversation_id: 'ServerDisconnects',
          receiver_id: world_id,
          data: {connection_id: connection_id},
        };
        _send_message(
          world_id,
          AGENT_DISCONNECT,
          msg,
        );
        active_connections.delete(connection_id);
      }
    }
  }

  // Re-call this thead, as it should run forever
  main_thread_timeout = setTimeout(main_thread, 50);
}

// ======================= </Threads> ======================

// ======================= <Routing> =======================

// Wrapper around getting the hit config details
function _load_hit_config() {
  var content = fs.readFileSync(task_directory_name + '/hit_config.json');
  return JSON.parse(content);
}

app.post('/sns_posts', async function(req, res, next) {
  res.end('Successful POST');
  let content;
  if (req.headers['x-amz-sns-message-type'] == 'SubscriptionConfirmation') {
    content = JSON.parse(req.body);
    var confirm_url = content.SubscribeURL;
    request(confirm_url, function(error, response, body) {
      if (!error && response.statusCode == 200) {
        console.log('Subscribed successfully');
      }
    });
  } else {
    var task_group_id = req.query['task_group_id'];
    content = JSON.parse(req.body);
    if (content['MessageId'] != '') {
      var message_id = content['MessageId'];
      var sender_id = 'AmazonMTurk';
      var message = JSON.parse(content['Message']);
      var event_type = message['Events'][0]['EventType'];
      var assignment_id = message['Events'][0]['AssignmentId'];
      var data = {
        text: event_type,
        id: sender_id,
        message_id: message_id,
      };
      var msg = {
        id: message_id,
        type: 'message',
        sender_id: sender_id,
        assignment_id: assignment_id,
        conversation_id: 'AmazonSNS',
        receiver_id: world_id,
        data: data,
      };
      _send_message(world_id, SNS_MESSAGE, msg);
      /// TODO on disconnects clean up static tasks (must be by assign id)
    }

  }
});

app.post('/submit_hit', async function(req, res, next) {
  res.end('Successful submit POST');
  let content = req.body;
  var task_group_id = content['task_group_id'];
  var agent_id = content['agent_id'];
  var assignment_id = content['assignment_id'];
  var worker_id = content['worker_id'];
  var response_data = content['response_data'];
  var message_id = assignment_id + '_' + worker_id + '_submit';

  var data = {
    text: '[DATA_SUBMIT]',
    id: agent_id,
    message_id: message_id,
    task_data: response_data,
  };
  var msg = {
    id: message_id,
    type: 'message',
    sender_id: worker_id,
    assignment_id: assignment_id,
    conversation_id: 'TaskSubmit',
    receiver_id: world_id,
    data: data,
  };
  _send_message(world_id, SUBMIT_MESSAGE, msg);
  /// TODO on submit clean up static tasks  (must be by assign id)
});

// Sometimes worker ids are corrupted in arriving from mturk for
// as of yet unknown reasons. We process those here
function fix_worker_id(worker_id) {
  if (worker_id) {
    // The only common issue is a worker id being comma-joined with itself.
    return worker_id.split(',')[0]
  }
  return worker_id
}

// Renders the chat page by setting up the template_context given the
// sent params for the request
app.get('/chat_index', async function(req, res) {
  var config_vars = _load_hit_config();
  var frame_height = config_vars.frame_height || 650;
  var allow_reviews = config_vars.allow_reviews || false;
  var block_mobile = config_vars.block_mobile;
  var chat_title = config_vars.chat_title || 'Live Chat';
  block_mobile = block_mobile === undefined ? true : block_mobile;
  var template_type = config_vars.template_type;

  var params = req.query;
  var template_context = {
    worker_id: fix_worker_id(params['workerId']),
    hit_id: params['hitId'],
    task_group_id: params['task_group_id'],
    assignment_id: params['assignmentId'],
    is_cover_page: params['assignmentId'] == 'ASSIGNMENT_ID_NOT_AVAILABLE',
    allow_reviews: allow_reviews,
    frame_height: frame_height,
    block_mobile: block_mobile,
    chat_title: chat_title,
    template_type: template_type,
  };

  res.render('index.html', template_context);
});

// Returns the hit config
app.get('/get_hit_config', function(req, res) {
  res.json(_load_hit_config());
});

// Returns server time for now
app.get('/get_timestamp', function(req, res) {
  res.json({ timestamp: Date.now() }); // in milliseconds
});

app.use(express.static('static'));

// ======================= </Routing> =======================
