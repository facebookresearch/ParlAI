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
const WebSocket = require('ws');

const task_directory_name = 'task';

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

// Start a socket to make a connection between the world and
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Socket used for speaking to the world
var world_socket = null;

// Handles sending a message through the socket
function _send_message(event_name, event_data) {
  if (world_socket) {
    var packet = {
      type: event_name,
      content: event_data,
    };
    world_socket.send(JSON.stringify(packet), function ack(error) {
      if (error === undefined) {
        return true;
      }
      console.log('Ran into error trying to send, retrying');
      setTimeout(function() {
        world_socket.send(JSON.stringify(packet), function ack2(error2) {
          if (error2 === undefined) {
            return true;
          }
          console.log('Repeat send of packet failed');
          console.log(packet);
          console.log(error2);
        });
      }, 500);
    });
    return true;
  } else {
    console.log('Message recieved without world connected');
    console.log(event_data);
    return false;
  }
}

// Register handlers
wss.on('connection', function(socket) {
  console.log('Client connected');
  // Disconnects are logged
  socket.on('disconnect', function() {
    console.log('World disconnected');
  });

  socket.on('message', function(data) {
    data = JSON.parse(data);
    if (data['type'] == 'world_alive') {
      world_socket = socket;
    } else if (data['type'] == 'ping') {
      socket.send(JSON.stringify({ type: 'pong', content: 'pong' }));
    }
  });

  socket.on('error', err => {
    console.log('Caught socket error');
    console.log(err);
  });

  socket.send(
    JSON.stringify({ type: 'conn_success', content: 'Socket is open!' }),
    function ack(error) {
      return;
    }
  );
});

server.listen(PORT, function() {
  console.log('Listening on %d', server.address().port);
});

// ----- Messenger routing functions -----
// Verify webhook
app.get('/webhook', async function(req, res) {
  // webhook verification token
  let VERIFY_TOKEN = 'Messenger4ParlAI';

  // Parse the query params
  let mode = req.query['hub.mode'];
  let token = req.query['hub.verify_token'];
  let challenge = req.query['hub.challenge'];

  // Checks if a token and mode is in the query string of the request
  if (mode && token) {
    // Checks the mode and token sent is correct
    if (mode === 'subscribe' && token === VERIFY_TOKEN) {
      // Responds with the challenge token from the request
      console.log('WEBHOOK_VERIFIED');
      res.status(200).send(challenge);
    } else {
      // Responds with '403 Forbidden' if verify tokens do not match
      res.sendStatus(403);
    }
  }
});

// Send messages through webhook
app.post('/webhook', async function(req, res, next) {
  let body = req.body;
  console.log(body);
  // Checks this is an event from a page subscription
  if (body.object === 'page') {
    try {
      let result = _send_message('new_packet', req.body);
      // TODO handle v. rare cases of message drops - should send timeout status
      if (result) {
        res.status(200).send('Successful POST');
      } else {
        res.status(504).send('Timeout');
      }
    } catch (error) {
      console.log('Transient error on message');
      console.log(error);
      console.log(req.body);
      res.status(504).send('Timeout');
    }
  } else {
    res.sendStatus(404);
  }
});
