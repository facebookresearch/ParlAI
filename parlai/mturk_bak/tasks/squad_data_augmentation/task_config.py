# Copyright 2004-present Facebook. All Rights Reserved.
# Task config for MTurk task

task_config = {}

# Task specific
task_config['teacher_agent_id'] = 'teacher'
task_config['worker_agent_id'] = task_config['teacher_agent_id']
task_config['bot_agent_id'] = 'recorder'

# Required for all tasks
task_config['agent_display_names'] = {
    # agent_id: display_name
    task_config['teacher_agent_id']: 'Teacher',
    task_config['bot_agent_id']: 'Recorder',
}
task_config['task_description'] = \
'''\'\'\'
(<b>Note</b>: You need to edit this text to suit your task.)<br><br>
You are going to chat with a bot regarding a particular topic.<br><br>
The responses you receive may not make much sense, but please give an appropriate evaluation on them or respond to them accordingly. 
\'\'\''''

# Only the initial_state can have no precondition
task_config['state_config'] = [
    {
        'state_name': 'initial_state',
        'precondition': None,
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'idle'
            }
        }
    },
    {
        'state_name': 'teacher_should_ask_question',
        'precondition': 'context',
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'text_input',
                'prompt_text': '''Please ask a question regarding this paragraph:''',
            }
        }
    },
    {
        'state_name': 'teacher_should_answer_question',
        'precondition': task_config['teacher_agent_id'],
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'text_input',
                'prompt_text': '''Please provide the answer to your question:'''
            }
        }
    },
    {
        'state_name': 'task_done',
        'precondition': task_config['teacher_agent_id'],
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'done',
            }
        }
    }
]

# 'response_config': {
    #   'response_type': 'choices',
    #   'prompt_text': '''Please select:''',
    #   'choices': [
    #       {
    #           'name': 'Choice 1',
    #           'display_name': 'Choice 1',
    #           'value': 1
    #       }
    #       {
    #           'name': 'Choice 2',
    #           'display_name': 'Choice 2',
    #           'value': 2
    #       }
    #       {
    #           'name': 'Choice 3',
    #           'display_name': 'Choice 3',
    #           'value': 3
    #       }
    #   ],
    # }

'''
idle
waiting -> prompt text
choices -> prompt text, list of choices
text_input -> prompt text
binary_reward -> prompt text
done -> prompt text
'''



