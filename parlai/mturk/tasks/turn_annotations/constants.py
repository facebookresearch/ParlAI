#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

AGENT_0 = 'Person1'
AGENT_1 = 'Person2'

WAITING_MSG = 'Please wait while we match you with another worker...'

ANNOTATIONS_CONFIG = [
    {
        'value': 'bucket_0',
        'name': 'Bucket 0',
        'description': 'this response implies something...0',
    },
    {
        'value': 'bucket_1',
        'name': 'Bucket 1',
        'description': 'this response implies something...1',
    },
    {
        'value': 'bucket_2',
        'name': 'Bucket 2',
        'description': 'this response implies something...2',
    },
    {
        'value': 'bucket_3',
        'name': 'Bucket 3',
        'description': 'this response implies something...3',
    },
    {
        'value': 'bucket_4',
        'name': 'Bucket 4',
        'description': 'this response implies something...4',
    },
]

ONBOARD_SUBMIT = '[ONBOARD_SUBMIT]'
ONBOARD_TRY_AGAIN = '[ONBOARD_TRY_AGAIN]'
ONBOARD_FAIL = '[ONBOARD_FAIL]'
ONBOARD_SUCCESS = '[ONBOARD_SUCCESS]'

ONBOARD_TASK_DATA = [
    {'text': 'Lorem ipsum dolor sit amet, unum dolor lobortis ad pro', 'agent_idx': 0},
    {
        'text': 'Lorem ipsum dolor sit amet, unum dolor lobortis ad pro',
        'agent_idx': 1,
        'answers': ['bucket_0'],
    },
    {'text': 'Lorem ipsum dolor sit amet, unum dolor lobortis ad pro', 'agent_idx': 0},
    {
        'text': 'Lorem ipsum dolor sit amet, unum dolor lobortis ad pro',
        'agent_idx': 1,
        'answers': ['bucket_1'],
    },
    {'text': 'Lorem ipsum dolor sit amet, unum dolor lobortis ad pro', 'agent_idx': 0},
    {
        'text': 'Lorem ipsum dolor sit amet, unum dolor lobortis ad pro',
        'agent_idx': 1,
        'answers': ['bucket_2'],
    },
    {'text': 'Lorem ipsum dolor sit amet, unum dolor lobortis ad pro', 'agent_idx': 0},
    {
        'text': 'Lorem ipsum dolor sit amet, unum dolor lobortis ad pro',
        'agent_idx': 1,
        'answers': ['bucket_3'],
    },
    {'text': 'Lorem ipsum dolor sit amet, unum dolor lobortis ad pro', 'agent_idx': 0},
    {
        'text': 'Lorem ipsum dolor sit amet, unum dolor lobortis ad pro',
        'agent_idx': 1,
        'answers': ['bucket_4'],
    },
]


"""
Task Instructions
"""

TASK_CONFIG = {}
TASK_CONFIG['hit_title'] = 'Chat with a fellow conversationalist!'
TASK_CONFIG['hit_description'] = 'Lorem Ipsum'

TASK_CONFIG['hit_keywords'] = 'chat,dialog'
TASK_CONFIG[
    'task_description'
] = '''
<br>
<b><h4>Task Description</h4></b>
<br>
Dummy Task Description.

Lorem ipsum.
<br><br>

'''

LEFT_PANE_TEXT = '''
<br>
<br>
<b><h4>Task Description</h4></b>
<br>
Dummy Left Pane Task Description. 

Lorem ipsum.
'''
