/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

const express = require('express');
const app = module.exports.app = express();
const path = require('path');
const server = require('http').createServer(app);
const PORT = 8080;
const open = require('open');

app.get('/bundle.js', function(req, res) {
  res.sendFile(path.join(__dirname, './build/bundle.js'));
});
app.get('*', function(req, res) {
  res.sendFile(path.join(__dirname, './src/static/index.html'));
});

server.listen(8080, function() {
  console.log("DialCrowd configuration tool running at http://localhost:" + PORT);
  open('http://localhost:' + PORT);
});
