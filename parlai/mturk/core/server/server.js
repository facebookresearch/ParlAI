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

nunjucks.configure('task', {
    autoescape: true,
    express: app
});

// ======================= Routing =======================

async function _is_invalid_HIT(task_group_id) {
  var hit_config = _load_hit_config();
  var max_allocation_count = hit_config['num_hits'] * hit_config['num_assignments'] * hit_config['mturk_agent_ids'].length;
  var total_allocation_count = await data_model.get_allocation_count(task_group_id);
  return (total_allocation_count === max_allocation_count);
}

async function _worker_did_refresh_HIT_page(task_group_id, assignment_id) {
  return await data_model.check_assignment_exists(task_group_id, assignment_id);
}

function _load_hit_config() {
  var content = fs.readFileSync(task_directory_name+'/hit_config.json');
  return JSON.parse(content);
}

function _load_server_config() {
  var content = fs.readFileSync('server_config.json');
  return JSON.parse(content);
}

//res.render('index.html', { foo: 'bar' });

// GET method route
app.get('/chat_index', async function (req, res) {
  var template_context = {};
  var params = req.query;

  var task_group_id = params['task_group_id'];
  var assignment_id = params['assignmentId']; // from mturk

  if (await _worker_did_refresh_HIT_page(task_group_id, assignment_id)) {
    res.send("Sorry, this HIT has expired because you closed or refreshed the page.");
  }
  else if (await _is_invalid_HIT(task_group_id)) {
    res.send("Sorry, all HITs in this group had already expired.");
  }
  else if (assignment_id === 'ASSIGNMENT_ID_NOT_AVAILABLE') {
    template_context['is_cover_page'] = true;
    res.render('cover_page.html', template_context);
  }
  else {
    var hit_config = _load_hit_config();

    var hit_id = params['hitId'];
    var worker_id = params['workerId'];

    var num_assignments = parseInt(hit_config['num_assignments']);
    var mturk_agent_ids = hit_config['mturk_agent_ids'];

    var hit_assignment_info = await data_model.sync_hit_assignment_info(
        task_group_id,
        num_assignments,
        mturk_agent_ids,
        assignment_id,
        hit_id,
        worker_id
    );

    var hit_index = hit_assignment_info['hit_index'];
    var assignment_index = hit_assignment_info['assignment_index'];
    var mturk_agent_id = hit_assignment_info['mturk_agent_id'];

    template_context['is_cover_page'] = false;
    template_context['task_group_id'] = task_group_id;
    template_context['hit_index'] = hit_index;
    template_context['assignment_index'] = assignment_index;
    template_context['conversation_id'] = hit_index + '_' + assignment_index;
    template_context['cur_agent_id'] = mturk_agent_id;
    template_context['frame_height'] = 650;

    // TODO: address custom template case
    /*
    custom_index_page = mturk_agent_id + '_index.html'
    if os.path.exists(custom_index_page):
        return render_template(custom_index_page, **template_context)
    else:
        return render_template('mturk_index.html', **template_context)
    */
    res.render('mturk_index.html', template_context);
  }
})

app.get('/get_hit_config', function (req, res) {
  res.json(_load_hit_config());
});

app.get('/get_hit_assignment_info', async function (req, res) {
  var params = req.query;
  var task_group_id = params['task_group_id'];
  var agent_id = params['agent_id'];
  var conversation_id = params['conversation_id'];

  res.json(await data_model.get_hit_assignment_info(task_group_id, agent_id, conversation_id));
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

// POST method route
// app.post('/', function (req, res) {
//   res.send('POST request to the homepage')
// })

// ======================= Routing =======================

// ======================= Socket =======================

const io = socketIO(
  app.listen(PORT, () => console.log(`Listening on ${ PORT }`))
);

var agent_global_id_to_room_id = {};
var room_id_to_agent_global_id = {};
var agent_global_id_to_event_name = {};
var agent_global_id_to_domain = {};

var commands_sent = {};
var messages_sent = {};

function get_agent_global_id(task_group_id, conversation_id, agent_id) {
  // Global ID within the server context
  if (agent_id === '[World]') {
    return agent_id;
  } else {
    return task_group_id + '_' + conversation_id + '_' + agent_id;
  }
}

function _send_event_to_agent(socket, task_group_id, conversation_id, agent_id, event_name, event_data, callback_function) {
  var agent_global_id = get_agent_global_id(task_group_id, conversation_id, agent_id);
  var agent_room_id = agent_global_id_to_room_id[agent_global_id];
  // Get agent domain
  var agent_domain = agent_global_id_to_domain[agent_global_id];
  if (!agent_room_id || !agent_domain) { // Server does not have information about this agent. Should wait for this agent's agent_alive event instead.
    return;
  }
  agent_domain.run(function(){
    lock.acquire(agent_global_id, function(){
      socket.to(agent_room_id).emit(event_name, event_data);
      waitUntil()
      .interval(500)
      .times(2)
      .condition(function() {
        console.log(agent_global_id+': Waiting for: '+event_name + '_received');
        return (agent_global_id_to_event_name[agent_global_id] === event_name + '_received');
      })
      .done(function(result) {
        if (result === false) { // Timeout
          console.log(agent_global_id+': Waiting for: '+event_name + '_received'+' failed. Resending event...');
          _send_event_to_agent(socket, task_group_id, conversation_id, agent_id, event_name, event_data, callback_function);
        } else { // Success
          console.log(agent_global_id+': Received: '+event_name + '_received');
          if (callback_function) {
            callback_function();
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
      var agent_global_id = room_id_to_agent_global_id[socket.id];
      console.log('Client disconnected: '+agent_global_id);
    });

    socket.on('agent_alive', (data, ack) => {
      console.log('on_agent_alive', data);
      var task_group_id = data['task_group_id'];
      var conversation_id = data['conversation_id'];
      var agent_id = data['agent_id'];

      var agent_global_id = get_agent_global_id(task_group_id, conversation_id, agent_id);
      agent_global_id_to_room_id[agent_global_id] = socket.id;
      room_id_to_agent_global_id[socket.id] = agent_global_id;
      if (!(agent_global_id_to_domain[agent_global_id])) {
        agent_global_id_to_domain[agent_global_id] = domain.create();
      }
      agent_global_id_to_event_name[agent_global_id] = 'agent_alive';

      if (!(agent_id === '[World]')) {
        _send_event_to_agent(
          socket,
          task_group_id, 
          null,
          '[World]',
          'agent_alive', 
          data,
          ack,
        );
      } else {
        ack(data);
      }
    });

    socket.on('agent_send_command', (data, ack) => {
      console.log('agent_send_command', data);

      var command_id = data['command_id'];
      var task_group_id = data['task_group_id'];
      var conversation_id = data['conversation_id'];
      var sender_agent_id = data['sender_agent_id'];
      var receiver_agent_id = data['receiver_agent_id'];
      var command = data['command'];

      if (commands_sent[command_id]) {
        ack();
        return;
      }

      var agent_global_id = get_agent_global_id(task_group_id, conversation_id, sender_agent_id);
      agent_global_id_to_event_name[agent_global_id] = 'agent_send_command';

      // var new_command_object = await data_model.send_new_command(
      //   task_group_id, 
      //   conversation_id,
      //   sender_agent_id,
      //   receiver_agent_id, 
      //   command
      // );

      var command_dict = {
        task_group_id: task_group_id,
        conversation_id: conversation_id,
        sender_agent_id: sender_agent_id,
        receiver_agent_id: receiver_agent_id,
        command: command,
        //command_id: new_command_object.id
        command_id: command_id
      }

      // Forward command to receiver agent.
      _send_event_to_agent(
        socket, 
        task_group_id, 
        conversation_id,
        receiver_agent_id,
        'new_command', 
        command_dict,
        function() {
          commands_sent[command_id] = true;
          // Send acknowledge event back to command sender.
          ack(command_dict);
        }
      );
    });

    socket.on('agent_send_message', (data, ack) => {
      console.log('agent_send_message', data);
      var message = data;
      var message_id = message['message_id'];
      var task_group_id = message['task_group_id'];
      var conversation_id = message['conversation_id'];
      var sender_agent_id = message['sender_agent_id'];
      var receiver_agent_id = message['receiver_agent_id'];
      var timestamp = message['timestamp'];
      var text = message['text'] || null;
      var reward = message['reward'] || null;
      var episode_done = message['episode_done'] || false;

      if (messages_sent[message_id]) {
        ack();
        return;
      }

      var agent_global_id = get_agent_global_id(task_group_id, conversation_id, sender_agent_id);
      agent_global_id_to_event_name[agent_global_id] = 'agent_send_command';

      // var new_message_object = await data_model.send_new_message(
      //   task_group_id, 
      //   conversation_id, 
      //   sender_agent_id, 
      //   receiver_agent_id,
      //   text, 
      //   reward,
      //   episode_done
      // );

      message = {
        task_group_id: task_group_id,
        conversation_id: conversation_id,
        sender_agent_id: sender_agent_id,
        id: sender_agent_id,
        receiver_agent_id: receiver_agent_id,
        episode_done: episode_done,
        //message_id: new_message_object['id'],
        message_id: message_id,
        timestamp: timestamp
      }
      if (text) message['text'] = text;
      if (reward) message['reward'] = reward;

      // Forward message to receiver agent.
      _send_event_to_agent(
        socket, 
        task_group_id, 
        conversation_id,
        receiver_agent_id,
        'new_message', 
        message,
        function() {
          messages_sent[message_id] = true;
          // Send acknowledge event back to message sender.
          ack(message);
        }
      );      
    });

    socket.on('new_command_received', (data, ack) => {
      var task_group_id = data['task_group_id'];
      var conversation_id = data['conversation_id'];
      var agent_id = data['agent_id'];

      var agent_global_id = get_agent_global_id(task_group_id, conversation_id, agent_id);
      agent_global_id_to_event_name[agent_global_id] = 'new_command_received';

      ack();
    });

    socket.on('new_message_received', (data, ack) => {
      var task_group_id = data['task_group_id'];
      var conversation_id = data['conversation_id'];
      var agent_id = data['agent_id'];

      var agent_global_id = get_agent_global_id(task_group_id, conversation_id, agent_id);
      agent_global_id_to_event_name[agent_global_id] = 'new_message_received';

      ack();
    });

    socket.on('agent_alive_received', (data, ack) => {
      var task_group_id = data['task_group_id'];
      var conversation_id = data['conversation_id'];
      var agent_id = data['agent_id'];

      var agent_global_id = get_agent_global_id(task_group_id, conversation_id, agent_id);
      agent_global_id_to_event_name[agent_global_id] = 'agent_alive_received';

      ack();
    });

    // Setup is done, ready to accept events.
    socket.emit('socket_open', 'Socket is open!');
});

// ======================= Socket =======================
