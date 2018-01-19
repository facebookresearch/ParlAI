/* Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
var request = require('request');

function getKey(req, res) {
  // Your verify token. Should be a random string.
  let VERIFY_TOKEN = "##VERIFY_TOKEN##"

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
}

function send_message(req, res) {
  let body = req.body;
  console.log(body);
  // Checks this is an event from a page subscription
  if (body.object === 'page') {
    request.post({
      url: '##HEROKU_URL##',
      body: body,
      json: true
    }, function (error, response, body) {
      if (error) {
        console.error(error);
        res.status(500).send('Internal Service Error');
      }
      res.status(response.statusCode).send('EVENT RECEIVED');
    });
  } else {
    res.sendStatus(404);
  }
}

exports.forward_message = function foward_message (req, res) {
  switch (req.method) {
    case 'GET':
      getKey(req, res);
      break;
    case 'POST':
      send_message(req, res);
      break;
    default:
      res.status(500).send('Unsupported operation');
      break;
  }
}
