# Copyright 2004-present Facebook. All Rights Reserved.
# Sample task config for MTurk task

# Task specific
teacher_agent_id = 'teacher'
worker_agent_id = teacher_agent_id
bot_agent_id = 'student'

# Required for all tasks
agent_display_names = {
    # agent_id: display_name
    teacher_agent_id: 'Teacher',
    bot_agent_id: 'Bot',
}
task_description = \
'''\'\'\'
(<b>Note</b>: You need to edit this text to suit your task.)<br><br>
You are going to chat with a bot regarding a particular topic.<br><br>
The responses you receive may not make much sense, but please give an appropriate evaluation on them or respond to them accordingly. 
\'\'\''''

# Only the initial_state can have no precondition
state_config = [
    {
        'state_name': 'initial_state',
        'precondition': None,
        'response_config': {
            worker_agent_id: {
                'response_type': 'idle'
            }
        }
    },
    {
        'state_name': 'teacher_should_ask_question',
        'precondition': worker_agent_id,
        'response_config': {
            worker_agent_id: {
                'response_type': 'text_input',
                'prompt_text': '''Please enter your question:''',
            }
        }
    },
    {
        'state_name': 'student_should_answer_question',
        'precondition': worker_agent_id,
        'response_config': {
            worker_agent_id: {
                'response_type': 'waiting',
                'prompt_text': '''Please wait for the other party's response...'''
            }
        }
    },
    {
        'state_name': 'teacher_should_give_reward',
        'precondition': bot_agent_id,
        'response_config': {
            worker_agent_id: {
                'response_type': 'binary_reward',
                'prompt_text': '''Please select reward:''',
                'reward_message_texts': {
                    'positive': 'This is correct.',
                    'negative': 'This is incorrect.',
                },
            }
        }
    },
    {
        'state_name': 'task_done',
        'precondition': worker_agent_id,
        'response_config': {
            worker_agent_id: {
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



