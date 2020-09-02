#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

AGENT_0 = 'Person1'
AGENT_1 = 'Person2'

WAITING_MSG = 'Please wait while we match you with another worker...'

ONBOARD_SUBMIT = '[ONBOARD_SUBMIT]'
ONBOARD_TRY_AGAIN = '[ONBOARD_TRY_AGAIN]'
ONBOARD_FAIL = '[ONBOARD_FAIL]'
ONBOARD_SUCCESS = '[ONBOARD_SUCCESS]'

ONBOARD_CONFIG = {
    'min_correct': 4,
    'max_incorrect': 3,
    'onboard_failures_max_allowed': 1,
}

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
