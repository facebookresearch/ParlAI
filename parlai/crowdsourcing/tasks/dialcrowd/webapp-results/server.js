/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

const express = require('express');
const app = module.exports.app = express();
const path = require('path');
const server = require('http').createServer(app);
const PORT = 5000;
const open = require('open');

var fs = require('fs')

/* Configure mephisto_path to your data path 
* Configure workers array to the data points you want to analyze
*/
var mephisto_path = 'PATH TO MEPHISTO DATA'
var workers = ['<task_run_id>/<assignment_id>/<agent_id>']

app.get('/bundle.js', function(req, res) {
  res.sendFile(path.join(__dirname, './build/bundle.js'));
});
app.get('/data', function(req, res){
  var sending_data = {data: []}
  workers.forEach((user, i) => {
    fs.readFile(mephisto_path + user + '/agent_data.json', 'utf8', function(err, agent_data){
      sending_data['data'].push({'userId': user.split('/').join('-'), 'mephisto_data': agent_data})
      if (sending_data['data'].length == workers.length){
        console.log(sending_data);
        res.send(sending_data);
      }
    })
  })
});
app.get('/config', function(req, res){
  fs.readFile('../task_config/config.json', 'utf8', function(err, data){
    res.send(data);
  })
})
app.get('*', function(req, res) {
  res.sendFile(path.join(__dirname, './src/static/index.html'));
});

server.listen(5000, function() {
  console.log("DialCrowd configuration tool running at http://localhost:" + PORT);
  open('http://localhost:' + PORT);
});