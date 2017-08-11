'use strict';

const express = require('express');
const socketIO = require('socket.io');
const nunjucks = require('nunjucks');
const path = require('path');
const fs = require("fs");
const domain = require('domain');
const AsyncLock = require('async-lock');
const bodyParser = require('body-parser');
const data_model = require("./data_model");

const task_directory_name = 'task'

const PORT = process.env.PORT || 3000;

const app = express()
app.use(bodyParser.json());

var lock = new AsyncLock();

nunjucks.configure(task_directory_name, {
    autoescape: true,
    express: app
});

// ======================= Routing =======================

function _load_hit_config() {
  var content = fs.readFileSync(task_directory_name+'/hit_config.json');
  return JSON.parse(content);
}

function _load_server_config() {
  var content = fs.readFileSync('server_config.json');
  return JSON.parse(content);
}

//res.render('index.html', { foo: 'bar' });

app.get('/chat_index', async function (req, res) {
  var template_context = {};
  var params = req.query;

  var assignment_id = params['assignmentId']; // from mturk
  var conversation_id = params['conversation_id'] || null;
  var mturk_agent_id = params['mturk_agent_id'] || null;
  var task_group_id = params['task_group_id'];
  var worker_id = params['workerId'] || null;
  var changing_conversation = params['changing_conversation'] || false;

  if (assignment_id === 'ASSIGNMENT_ID_NOT_AVAILABLE') {
    template_context['is_cover_page'] = true;
    res.render('cover_page.html', template_context);
  }
  else if ((!changing_conversation) && _load_hit_config()['unique_worker'] === true && await data_model.worker_record_exists(task_group_id, worker_id)) {
    res.send("Sorry, but you can only work on this HIT once.");
  }
  else {
    await data_model.add_worker_record(task_group_id, worker_id);
    if (!conversation_id && !mturk_agent_id) { // if conversation info is not loaded yet
      // TODO: change to a loading indicator
      template_context['is_init_page'] = true;
      res.render('mturk_index.html', template_context);
    }
    else {
      var hit_id = params['hitId'];

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

app.get('/clean_database', function (req, res) {
  var params = req.query;
  var db_host = _load_server_config()['db_host'];

  if (params['db_host'] === db_host) {
    data_model.clean_database();
    res.sendStatus(200);
  } else {
    res.sendStatus(401);
  }
});

app.get('/get_timestamp', function (req, res) {
  res.json({'timestamp': Date.now()}); // in milliseconds
});

// ======================= Routing =======================

// ======================= Socket =======================

const io = socketIO(
  app.listen(PORT, () => console.log(`Listening on ${ PORT }`))
);

var worker_id_to_room_id = {};
var room_id_to_worker_id = {};

function _send_message(socket, worker_id, event_name, event_data) {
  var worker_room_id = worker_id_to_room_id[worker_id];
  // Server does not have information about this worker. Should wait for this worker's agent_alive event instead.
  if (!worker_room_id) {
    console.log("Worker room id doesn't exist! Skipping message.")
    return;
  }
  console.log('worker_room_id' + worker_room_id);
  socket.broadcast.in(worker_room_id).emit(event_name, event_data);
}

io.on('connection', function (socket) {
  console.log('Client connected');

  socket.on('disconnect', function () {
    var worker_id = room_id_to_worker_id[socket.id];
    console.log('Client disconnected: '+worker_id);

  });

  socket.on('agent alive', function (data, ack) {
    var worker_id = data["sender_id"];
    console.log('agent alive', data);
    worker_id_to_room_id[worker_id] = socket.id;
    room_id_to_worker_id[socket.id] = worker_id;

    if (!(worker_id === '[World]')) {
      _send_message(socket, data['receiver_id'], 'new message', data);
    }
    if(ack) {
        ack('agent_alive');
    }
  });

  socket.on('route message', function (data, ack) {
    console.log('route message', data);
    var worker_id = data['receiver_id'];

    _send_message(socket, worker_id, 'new message', data);
    if(ack) {
        ack('route message');
    }
  });

  socket.emit('socket_open', 'Socket is open!');
});

// ======================= Socket =======================
