# Copyright 2004-present Facebook. All Rights Reserved.
# Task config for MTurk task

task_config = {}

# MTurk config related
task_config['hit_title'] = 'Chat with a fellow Turker about a paragraph'
task_config['hit_description'] = 'Ask and answer a question about a paragraph with a fellow Turker.'
task_config['hit_keywords'] = 'chat,question,answer'
task_config['hit_reward'] = 0.05

# Task specific
task_config['teacher_agent_id'] = 'teacher'
task_config['worker_agent_id'] = task_config['teacher_agent_id']

# Required for all tasks
task_config['task_description'] = \
'''\'\'\'
(<b>Note</b>: You need to edit this text to suit your task.)<br><br>
You are going to chat with a bot regarding a particular topic.<br><br>
The responses you receive may not make much sense, but please give an appropriate evaluation on them or respond to them accordingly. 
\'\'\''''


