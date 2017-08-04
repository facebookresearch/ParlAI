'use strict';

const express = require('express');
const socketIO = require('socket.io');
const nunjucks = require('nunjucks');
const path = require('path');
const fs = require("fs");
const domain = require('domain');
const AsyncLock = require('async-lock');
const waitUntil = require('wait-until');
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
var worker_id_to_event_name = {};
var worker_id_to_event_data = {};
var worker_id_to_domain = {};

function _send_event_to_agent(socket, worker_id, event_name, event_data, callback_function) {
  var worker_room_id = worker_id_to_room_id[worker_id];
  // Get worker domain
  var worker_domain = worker_id_to_domain[worker_id];
  if (!worker_room_id || !worker_domain) { // Server does not have information about this worker. Should wait for this worker's agent_alive event instead.
    if (!worker_room_id) console.log("worker room id doesn't exist!");
    if (!worker_domain) console.log("worker domain doesn't exist!");
    return;
  }
  worker_domain.run(function(){
    lock.acquire(worker_id, function(){
      socket.to(worker_room_id).emit(event_name, event_data);
      waitUntil()
      .interval(500)
      .times(2)
      .condition(function() {
        console.log(worker_id+': Waiting for: '+event_name + '_received');
        return (worker_id_to_event_name[worker_id] === event_name + '_received');
      })
      .done(function(result) {
        if (result === false) { // Timeout
          console.log(worker_id+': Waiting for: '+event_name + '_received'+' failed. Resending event...');
          _send_event_to_agent(socket, worker_id, event_name, event_data, callback_function);
        } else { // Success
          console.log(worker_id+': Received: '+event_name + '_received');
          if (callback_function) {
            callback_function(worker_id_to_event_data[worker_id]);
          }
        }
      });
    }, function(err, ret){
      // lock released
    });
  });
}

io.on('connection', (socket) => {
    console.log('Client connected');

    socket.on('disconnect', () => {
      var worker_id = room_id_to_worker_id[socket.id];
      console.log('Client disconnected: '+worker_id);
    });

    socket.on('agent_alive', async (data, ack) => {
      console.log('on_agent_alive', data);
      var worker_id = data['worker_id'];

      worker_id_to_room_id[worker_id] = socket.id;
      room_id_to_worker_id[socket.id] = worker_id;
      if (!(worker_id_to_domain[worker_id])) {
        worker_id_to_domain[worker_id] = domain.create();
      }
      worker_id_to_event_name[worker_id] = 'agent_alive';
      worker_id_to_event_data[worker_id] = data;

      if (!(worker_id === '[World]')) {
        _send_event_to_agent(
          socket,
          '[World]',
          'agent_alive', 
          data,
          function(callback_data) {
            ack(callback_data);
          }
        );
      } else {
        ack(data);
      }
    });

    socket.on('agent_send_command', (data, ack) => {
      console.log('agent_send_command', data);

      var receiver_worker_id = data['receiver_worker_id'];
      var command_id = data['command_id'];

      worker_id_to_event_name['[World]'] = 'agent_send_command';
      worker_id_to_event_data['[World]'] = data;

      // Forward command to receiver agent.
      _send_event_to_agent(
        socket, 
        receiver_worker_id,
        'new_command', 
        data,
        function() {
          // Send acknowledge event back to command sender.
          ack(data);
        }
      );
    });

    socket.on('agent_send_message', (data, ack) => {
      console.log('agent_send_message', data);

      var receiver_worker_id = data['receiver_worker_id'];
      var message_id = data['message_id'];

      worker_id_to_event_name['[World]'] = 'agent_send_message';
      worker_id_to_event_data['[World]'] = data;

      // Forward message to receiver agent.
      _send_event_to_agent(
        socket, 
        receiver_worker_id,
        'new_message', 
        data,
        function() {
          // Send acknowledge event back to message sender.
          ack(data);
        }
      );      
    });

    socket.on('new_command_received', (data, ack) => {
      var by_worker_id = data['by_worker_id'];

      worker_id_to_event_name[by_worker_id] = 'new_command_received';
      worker_id_to_event_data[by_worker_id] = data;

      ack();
    });

    socket.on('new_message_received', (data, ack) => {
      var by_worker_id = data['by_worker_id'];

      worker_id_to_event_name[by_worker_id] = 'new_message_received';
      worker_id_to_event_data[by_worker_id] = data;

      ack();
    });

    socket.on('agent_alive_received', (data, ack) => {
      var by_worker_id = data['by_worker_id'];

      worker_id_to_event_name[by_worker_id] = 'agent_alive_received';
      worker_id_to_event_data[by_worker_id] = data;

      ack();
    });

    // Setup is done, ready to accept events.
    socket.emit('socket_open', 'Socket is open!');
});

// ======================= Socket =======================
