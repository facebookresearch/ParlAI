# Copyright 2004-present Facebook. All Rights Reserved.
# Task config for MTurk task

task_config = {}

# MTurk config related
task_config['hit_title'] = 'Chat with a fellow Turker about a Wikipedia paragraph'
task_config['hit_description'] = 'Ask and answer a question about a Wikipedia paragraph with a fellow Turker.'
task_config['hit_keywords'] = 'chat,question,answer'
task_config['hit_reward'] = 0.10

# Task specific
task_config['teacher_agent_id'] = 'teacher'
task_config['worker_agent_id'] = task_config['teacher_agent_id']
task_config['bot_agent_id'] = 'student'

# Required for all tasks
task_config['agent_display_names'] = {
    # agent_id: display_name
    task_config['teacher_agent_id']: 'Teacher',
    task_config['bot_agent_id']: 'Student',
}
task_config['task_description'] = \
'''\'\'\'
(<b>Note</b>: You need to edit this text to suit your task.)<br><br>
You are going to chat with a bot regarding a particular topic.<br><br>
The responses you receive may not make much sense, but please give an appropriate evaluation on them or respond to them accordingly. 
\'\'\''''

"""
Response type accepted:
idle
waiting -> prompt text
choices -> prompt text, list of choices
text_input -> prompt text
binary_reward -> prompt text, reward_message_texts
done -> prompt text
"""
# Only the initial_state can have no precondition
task_config['state_config'] = {
    'initial_state':
    {
        'state_name': 'initial_state',
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'idle'
            }
        },
        'next_states': [
            {
                'precondition_agent_id': 'context',
                'precondition_agent_action': 'any',
                'state_name': 'teacher_should_ask_question'
            }
        ]
    },
    'teacher_should_ask_question':
    {
        'state_name': 'teacher_should_ask_question',
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'idle'
            }
        },
        'next_states': [
            {
                'precondition_agent_id': task_config['teacher_agent_id'],
                'precondition_agent_action': 'any',
                'state_name': 'student_should_answer_question'
            }
        ]
    },
    'student_should_answer_question':
    {
        'state_name': 'student_should_answer_question',
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'waiting',
                'prompt_text': '''Please wait for the bot's response...'''
            }
        },
        'next_states': [
            {
                'precondition_agent_id': task_config['bot_agent_id'],
                'precondition_agent_action': 'any',
                'state_name': 'teacher_should_give_textual_feedback'
            }
        ]
    },
    'teacher_should_give_textual_feedback':
    {
        'state_name': 'teacher_should_give_textual_feedback',
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'text_input',
                'prompt_text': '''Please provide textual feedback regarding whether the student's answer is correct. Feel free to express this in any form.'''
            }
        },
        'next_states': [
            {
                'precondition_agent_id': task_config['teacher_agent_id'],
                'precondition_agent_action': 'any',
                'state_name': 'teacher_should_give_reward'
            }
        ]
    },
    'teacher_should_give_reward':
    {
        'state_name': 'teacher_should_give_reward',
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'binary_reward',
                'prompt_text': '''Now please express this in another form: Do you think the student's answer is correct?''',
            }
        },
        'next_states': [
            {
                'precondition_agent_id': task_config['teacher_agent_id'],
                'precondition_agent_action': 'correct',
                'state_name': 'task_done'
            },
            {
                'precondition_agent_id': task_config['teacher_agent_id'],
                'precondition_agent_action': 'incorrect',
                'state_name': 'teacher_should_provide_correct_answer'
            }
        ]
    },
    'teacher_should_provide_correct_answer':
    {
        'state_name': 'teacher_should_provide_correct_answer',
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'text_input',
                'prompt_text': '''Please enter the correct answer to the question:'''
            }
        },
        'next_states': [
            {
                'precondition_agent_id': task_config['teacher_agent_id'],
                'precondition_agent_action': 'any',
                'state_name': 'task_done'
            }
        ]
    },
    'task_done':
    {
        'state_name': 'task_done',
        'response_config': {
            task_config['teacher_agent_id']: {
                'response_type': 'done',
            }
        }
    }
}

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




