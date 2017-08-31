/* Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
'use strict';

const bodyParser = require('body-parser');
const express = require('express');
const fs = require("fs");
const nunjucks = require('nunjucks');
const socketIO = require('socket.io');

const task_directory_name = 'task'

const PORT = process.env.PORT || 3000;

// Initialize app
const app = express()
app.use(bodyParser.json());

nunjucks.configure(task_directory_name, {
    autoescape: true,
    express: app
});

// ======================= <Routing> =======================

// Wrapper around getting the hit config details
function _load_hit_config() {
  var content = fs.readFileSync(task_directory_name+'/hit_config.json');
  return JSON.parse(content);
}

// Renders the chat page by setting up the template_context given the
// sent params for the request
app.get('/chat_index', async function (req, res) {
  var template_context = {};
  var params = req.query;

  var assignment_id = params['assignmentId']; // from mturk
  var conversation_id = params['conversation_id'] || null;
  var mturk_agent_id = params['mturk_agent_id'] || null;

  if (assignment_id === 'ASSIGNMENT_ID_NOT_AVAILABLE') {
    // Render the cover page
    template_context['is_cover_page'] = true;
    res.render('cover_page.html', template_context);
  } else {
    if (!conversation_id && !mturk_agent_id) {
      // if conversation info is not loaded yet, go to an init page
      template_context['is_init_page'] = true;
      res.render('mturk_index.html', template_context);
    }
    else {
      // Set up template params
      template_context['is_cover_page'] = false;
      // TODO move this 650 to be in one location and one location only, it's
      // a magic number in multiple places
      template_context['frame_height'] = 650;
      template_context['cur_agent_id'] = mturk_agent_id;
      template_context['conversation_id'] = conversation_id;

      // Load custom pages by the mturk_agent_id if the custom pages exist
      var custom_index_page = mturk_agent_id + '_index.html';
      if (fs.existsSync(task_directory_name+'/'+custom_index_page)) {
        res.render(custom_index_page, template_context);
      } else {
        res.render('mturk_index.html', template_context);
      }
    }
  }
});

// Returns the hit config
app.get('/get_hit_config', function (req, res) {
  res.json(_load_hit_config());
});

// Returns server time for now
app.get('/get_timestamp', function (req, res) {
  res.json({'timestamp': Date.now()}); // in milliseconds
});

// ======================= </Routing> =======================

// ======================= <Socket> =======================

// Start a socket
const io = socketIO(
  app.listen(PORT, () => console.log(`Listening on ${ PORT }`))
);

// Track connections
var connection_id_to_room_id = {};
var room_id_to_connection_id = {};

// Handles sending a message through the socket
function _send_message(socket, connection_id, event_name, event_data) {
  // Find the room the connection exists in
  var connection_room_id = connection_id_to_room_id[connection_id];
  // Server does not have information about this worker. Should wait for this
  // worker's agent_alive event instead.
  if (!connection_room_id) {
    console.log('Connection room id for ' + connection_id +
      ' doesn\'t exist! Skipping message.')
    return;
  }
  // Send the message through
  socket.broadcast.in(connection_room_id).emit(event_name, event_data);
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

// Register handlers
io.on('connection', function (socket) {
  console.log('Client connected');

  // Disconnects are logged
  socket.on('disconnect', function () {
    var connection_id = room_id_to_connection_id[socket.id];
    console.log('Client disconnected: ' + connection_id);
  });

  // Agent alive events are handled by registering the agent to a connection_id
  // and then forwarding the alive to the world if it came from a client
  socket.on('agent alive', function (data, ack) {
    var sender_id = data['sender_id'];
    var in_connection_id = _get_from_conn_id(data);
    var out_connection_id = _get_to_conn_id(data);
    console.log('agent alive', data);
    connection_id_to_room_id[in_connection_id] = socket.id;
    room_id_to_connection_id[socket.id] = in_connection_id;
    console.log('connection_id ' + in_connection_id + ' registered');

    // Send alive packets to the world, but not from the world
    if (!(sender_id && sender_id.startsWith('[World'))) {
      _send_message(socket, out_connection_id, 'new packet', data);
    }
    // Acknowledge that the message was recieved
    if(ack) {
      ack('agent_alive');
    }
  });

  // handles routing a packet to the desired recipient
  socket.on('route packet', function (data, ack) {
    console.log('route packet', data);
    var out_connection_id = _get_to_conn_id(data);

    _send_message(socket, out_connection_id, 'new packet', data);
    // Acknowledge if required
    if(ack) {
      ack('route packet');
    }
  });

  socket.emit('socket_open', 'Socket is open!');
});

// ======================= </Socket> =======================
