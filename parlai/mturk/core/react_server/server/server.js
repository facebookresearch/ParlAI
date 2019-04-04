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

// ======================= <Socket> =======================

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Track connections
var connection_id_to_socket = {};
var room_id_to_connection_id = {};
var NOTIF_ID = 'MTURK_NOTIFICATIONS';

// Handles sending a message through the socket
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
  var reciever_id = data['receiver_id'];
  if (reciever_id && reciever_id.startsWith('[World')) {
    return reciever_id;
  } else {
    return reciever_id + '_' + data['assignment_id'];
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

function handle_route(data) {
  var out_connection_id = _get_to_conn_id(data);

  _send_message(out_connection_id, 'new packet', data);
}

function handle_pong(data) {
  data['type'] = 'pong';
  var out_connection_id = _get_from_conn_id(data);

  _send_message(out_connection_id, 'new packet', data);
}

// Agent alive events are handled by registering the agent to a connection_id
// and then forwarding the alive to the world if it came from a client
function handle_alive(socket, data) {
  var sender_id = data['sender_id'];
  var in_connection_id = _get_from_conn_id(data);
  var out_connection_id = _get_to_conn_id(data);
  connection_id_to_socket[in_connection_id] = socket;
  room_id_to_connection_id[socket.id] = in_connection_id;

  // Send alive packets to the world, but not from the world
  if (!(sender_id && sender_id.startsWith('[World'))) {
    _send_message(out_connection_id, 'new packet', data);
  } else {
    socket.send(
      JSON.stringify({ type: 'conn_success', content: 'Socket is open!' })
    );
  }
}

// Register handlers
wss.on('connection', function(socket) {
  console.log('Client connected');
  // Disconnects are logged
  socket.on('disconnect', function() {
    var connection_id = room_id_to_connection_id[socket.id];
    console.log('Client disconnected: ' + connection_id);
  });

  socket.on('error', err => {
    console.log('Caught socket error');
    console.log(err);
  });

  // handles routing a packet to the desired recipient
  socket.on('message', function(data) {
    try {
      data = JSON.parse(data);
      if (data['type'] == 'agent alive') {
        handle_alive(socket, data['content']);
      } else if (data['type'] == 'route packet') {
        handle_route(data['content']);
        if (data['content']['type'] == 'heartbeat') {
          handle_pong(data['content']);
        }
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
    var world_id = '[World_' + task_group_id + ']';
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
      _send_message(world_id, 'new packet', msg);
    }
  }
});


app.post('/submit_static', async function(req, res, next) {
  res.end('Successful static POST');
  let content = req.body;
  var task_group_id = content['task_group_id'];
  var agent_id = content['agent_id'];
  var assignment_id = content['assignment_id'];
  var worker_id = content['worker_id'];
  var response_data = content['response_data'];
  var world_id = '[World_' + task_group_id + ']';
  var message_id = assignment_id + '_' + worker_id + '_static_submit';

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
  _send_message(world_id, 'new packet', msg);
});

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
    worker_id: params['workerId'],
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
