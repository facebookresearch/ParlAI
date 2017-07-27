const Sequelize = require('sequelize');
const fs = require("fs");

var exports = module.exports = {};

const db_host = _load_server_config()['db_host'];
const db_name = _load_server_config()['db_name'];
const db_username = _load_server_config()['db_username'];
const db_password = _load_server_config()['db_password'];

const COMMAND_SEND_MESSAGE = 'COMMAND_SEND_MESSAGE'; // MTurk web client is expected to send a new message to server
const COMMAND_SHOW_DONE_BUTTON = 'COMMAND_SHOW_DONE_BUTTON'; // MTurk web client should show the "DONE" button
const COMMAND_EXPIRE_HIT = 'COMMAND_EXPIRE_HIT'; // MTurk web client should show "HIT is expired"

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

const MTurkHITAgentAllocation = sequelize.define('mturk_hit_agent_allocation', {
  task_group_id: { type: Sequelize.STRING },
  conversation_id: { type: Sequelize.STRING, allowNull: true, defaultValue: null },
  agent_id: { type: Sequelize.STRING, allowNull: true, defaultValue: null},
  assignment_id: { type: Sequelize.STRING, allowNull: true, defaultValue: null},
  hit_id: { type: Sequelize.STRING, allowNull: true, defaultValue: null},
  worker_id: { type: Sequelize.STRING, allowNull: true, defaultValue: null}
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

exports.sync_hit_assignment_info = async function(task_group_id, num_assignments, mturk_agent_ids, assignment_id, hit_id, worker_id) {
    if (mturk_agent_ids) {
        var new_allocation_object = await MTurkHITAgentAllocation.create({
                                                task_group_id: task_group_id,
                                                agent_id: null,
                                                conversation_id: null,
                                                assignment_id: assignment_id,
                                                hit_id: hit_id,
                                                worker_id: worker_id
                                            });
        var object_id = new_allocation_object.id;

        var existing_allocation_object_list = await MTurkHITAgentAllocation.findAll({
                                                    where: { task_group_id: task_group_id },
                                                    order: sequelize.col('id')
                                                });
        var existing_allocation_id_list = [];

        existing_allocation_object_list.forEach((allocation_object) => {
            existing_allocation_id_list.push(allocation_object.get('id'));
        });

        var index_in_list = existing_allocation_id_list.indexOf(object_id);

        if (index_in_list > -1) {
            var mturk_agent_id = mturk_agent_ids[index_in_list % mturk_agent_ids.length];
            var assignment_index = Math.trunc(Math.floor(index_in_list / mturk_agent_ids.length) % num_assignments + 1);
            var hit_index = Math.trunc(Math.floor(index_in_list / (mturk_agent_ids.length * num_assignments)) + 1);

            var conversation_id = hit_index + '_' + assignment_index;

            await MTurkHITAgentAllocation.update(
                {
                    conversation_id: conversation_id,
                    agent_id: mturk_agent_id
                },
                { 
                    where: { id: new_allocation_object.id }
                }
            );

            return {
                hit_index: hit_index, 
                assignment_index: assignment_index,
                mturk_agent_id: mturk_agent_id
            }
        }
    }
}

exports.get_hit_assignment_info = async function(task_group_id, agent_id, conversation_id) {
    var existing_allocation_object = await MTurkHITAgentAllocation.findOne({
                                                where: { 
                                                    task_group_id: task_group_id,
                                                    agent_id: agent_id,
                                                    conversation_id: conversation_id
                                                },
                                            });
    var assignment_id = null;
    var hit_id = null;
    var worker_id = null;

    if (existing_allocation_object) {
        existing_allocation_object = existing_allocation_object.get({plain: true});
        assignment_id = existing_allocation_object.assignment_id;
        hit_id = existing_allocation_object.hit_id;
        worker_id = existing_allocation_object.worker_id;
    }

    return {
        assignment_id: assignment_id,
        hit_id: hit_id,
        worker_id: worker_id
    }
}

exports.get_allocation_count = async function(task_group_id) {
    return await MTurkHITAgentAllocation.count({
                    where: { 
                        task_group_id: task_group_id
                    },
                });
}

exports.check_assignment_exists = async function(task_group_id, assignment_id) {
    var assignment_count = await MTurkHITAgentAllocation.count({
                                    where: { 
                                        task_group_id: task_group_id,
                                        assignment_id: assignment_id
                                    },
                                });
    return (assignment_count > 0);
}