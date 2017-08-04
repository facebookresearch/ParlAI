const Sequelize = require('sequelize');
const fs = require("fs");

var exports = module.exports = {};

const db_host = _load_server_config()['db_host'];
const db_name = _load_server_config()['db_name'];
const db_username = _load_server_config()['db_username'];
const db_password = _load_server_config()['db_password'];

var opts = {
    define: {
        //prevent Sequelize from pluralizing table names
        freezeTableName: true,
        timestamps: false
    }
}

function _load_server_config() {
  var content = fs.readFileSync('server_config.json');
  return JSON.parse(content);
}

const sequelize = new Sequelize('postgres://'+db_username+':'+db_password+'@'+db_host+':5432/'+db_name, opts);

sequelize
  .authenticate()
  .then(() => {
    console.log('Database connection has been established successfully.');
  })
  .catch(err => {
    console.error('Unable to connect to the database:', err);
  });

// ========================= Models =========================

// const Message = sequelize.define('message', {
//   task_group_id: { type: Sequelize.STRING },
//   conversation_id: { type: Sequelize.STRING },
//   sender_agent_id: { type: Sequelize.STRING },
//   receiver_agent_id: { type: Sequelize.STRING },
//   message_content: { type: Sequelize.TEXT }
// });

// const Command = sequelize.define('command', {
//   task_group_id: { type: Sequelize.STRING },
//   conversation_id: { type: Sequelize.STRING },
//   sender_agent_id: { type: Sequelize.STRING },
//   receiver_agent_id: { type: Sequelize.STRING },
//   command: { type: Sequelize.STRING }
// });

const MTurkWorkerRecord = sequelize.define('mturk_worker_record', {
    task_group_id: { type: Sequelize.STRING },
    worker_id: { type: Sequelize.STRING }
});

// ========================= Models =========================

// ========================= DB Management ==========================

exports.init_database = function() {
    sequelize.sync();
}

exports.clean_database = function() {
    sequelize.sync({force: true});
}

// exports.send_new_command = async function(task_group_id, conversation_id, sender_agent_id, receiver_agent_id, command) {
//     var new_command_object = await Command.create({
//                                     task_group_id: task_group_id,
//                                     conversation_id: conversation_id,
//                                     sender_agent_id: sender_agent_id,
//                                     receiver_agent_id: receiver_agent_id,
//                                     command: command
//                                 });
//     return new_command_object.get({plain: true});
// }

// exports.send_new_message = async function(task_group_id, conversation_id, sender_agent_id, receiver_agent_id, message_text, reward, episode_done) {
//     var new_message = {
//         text: message_text,
//         id: sender_agent_id,
//         episode_done: episode_done
//     }
//     if (reward) new_message['reward'] = reward;

//     var message_content = JSON.stringify(new_message);

//     var new_message_object = await Message.create({
//                                     task_group_id: task_group_id,
//                                     conversation_id: conversation_id,
//                                     sender_agent_id: sender_agent_id,
//                                     receiver_agent_id: receiver_agent_id,
//                                     message_content: message_content
//                                 });
//     return new_message_object.get({plain: true});
// }

exports.add_worker_record = async function(task_group_id, worker_id) {
    var existing_record_count = await MTurkWorkerRecord.count({
                                    where: { 
                                        task_group_id: task_group_id,
                                        worker_id: worker_id
                                    },
                                });
    if (existing_record_count === 0) {
        var new_record_object = await MTurkWorkerRecord.create({
                                        task_group_id: task_group_id,
                                        worker_id: worker_id
                                    });
    }
}

exports.worker_record_exists = async function(task_group_id, worker_id) {
    var existing_record_count = await MTurkWorkerRecord.count({
                                    where: { 
                                        task_group_id: task_group_id,
                                        worker_id: worker_id
                                    },
                                });
    return (existing_record_count > 0);
}