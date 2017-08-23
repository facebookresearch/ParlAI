'use strict';

const express = require('express');
const socketIO = require('socket.io');
const nunjucks = require('nunjucks');
const path = require('path');
const fs = require("fs");
const domain = require('domain');
const AsyncLock = require('async-lock');
const bodyParser = require('body-parser');
//const data_model = require("./data_model");

const task_directory_name = 'task'

const PORT = process.env.PORT || 3000;

const app = express()
app.use(bodyParser.json());

var lock = new AsyncLock();

nunjucks.configure(task_directory_name, {
    autoescape: true,
    express: app
});

// ======================= <Routing> =======================

function _load_hit_config() {
  var content = fs.readFileSync(task_directory_name+'/hit_config.json');
  return JSON.parse(content);
}

app.get('/chat_index', async function (req, res) {
  var template_context = {};
  var params = req.query;

  var assignment_id = params['assignmentId']; // from mturk
  var hit_id = params['hitId'];
  var conversation_id = params['conversation_id'] || null;
  var mturk_agent_id = params['mturk_agent_id'] || null;
  var task_group_id = params['task_group_id'];
  var worker_id = params['workerId'] || null;
  var changing_conversation = params['changing_conversation'] || false;

  if (assignment_id === 'ASSIGNMENT_ID_NOT_AVAILABLE') {
    template_context['is_cover_page'] = true;
    res.render('cover_page.html', template_context);
  } else {
    if (!conversation_id && !mturk_agent_id) {
      // if conversation info is not loaded yet
      // TODO: change to a loading indicator
      template_context['is_init_page'] = true;
      res.render('mturk_index.html', template_context);
    }
    else {
      template_context['is_cover_page'] = false;
      template_context['frame_height'] = 650;
      template_context['cur_agent_id'] = mturk_agent_id;
      template_context['conversation_id'] = conversation_id;

      var custom_index_page = mturk_agent_id + '_index.html';
      if (fs.existsSync(task_directory_name+'/'+custom_index_page)) {
        res.render(custom_index_page, template_context);
      } else {
        res.render('mturk_index.html', template_context);
      }
    }
  }
});

app.get('/get_hit_config', function (req, res) {
  res.json(_load_hit_config());
});

app.get('/get_timestamp', function (req, res) {
  res.json({'timestamp': Date.now()}); // in milliseconds
});

// ======================= </Routing> =======================

// ======================= <Socket> =======================

const io = socketIO(
  app.listen(PORT, () => console.log(`Listening on ${ PORT }`))
);

var connection_id_to_room_id = {};
var room_id_to_connection_id = {};

function _send_message(socket, connection_id, event_name, event_data) {
  var connection_room_id = connection_id_to_room_id[connection_id];
  // Server does not have information about this worker. Should wait for this
  // worker's agent_alive event instead.
  if (!connection_room_id) {
    console.log('Connection room id for ' + connection_id +
      ' doesn\'t exist! Skipping message.')
    return;
  }
  socket.broadcast.in(connection_room_id).emit(event_name, event_data);
}

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

io.on('connection', function (socket) {
  console.log('Client connected');

  socket.on('disconnect', function () {
    var connection_id = room_id_to_connection_id[socket.id];
    console.log('Client disconnected: ' + connection_id);
  });

  socket.on('agent alive', function (data, ack) {
    var sender_id = data['sender_id'];
    var in_connection_id = _get_from_conn_id(data);
    var out_connection_id = _get_to_conn_id(data);
    console.log('agent alive', data);
    connection_id_to_room_id[in_connection_id] = socket.id;
    room_id_to_connection_id[socket.id] = in_connection_id;
    console.log('connection_id ' + in_connection_id + ' registered');

    if (!(sender_id && sender_id.startsWith('[World'))) {
      // Send alive packets to the world, but not from the world
      _send_message(socket, out_connection_id, 'new packet', data);
    }
    if(ack) {
      ack('agent_alive');
    }
  });

  socket.on('route packet', function (data, ack) {
    console.log('route packet', data);
    var out_connection_id = _get_to_conn_id(data);

    _send_message(socket, out_connection_id, 'new packet', data);
    if(ack) {
        ack('route packet');
    }
  });

  socket.emit('socket_open', 'Socket is open!');
});

// ======================= </Socket> =======================
