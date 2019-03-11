#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}


task_config['frontend_version'] = 1

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Ask, answer, and evaluate numeric questions'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = \
    'Perform one of three roles to get accurate numeric questions.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,question,answer'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = '''
In this task, you'll need to ask, answer, or evaluate numeric questions.<br><br>
<b>As the asker:</b> please only ask questions that can be answered with a
number. Prefer to use natural language questions rather than typed
up equations.<br>
<b>As the answerer:</b> feel free to use external resources to come up with
answers to questions if you don't know the answer.<br>
<b>As the evaluator:</b> please evaluate based on accuracy of the answers as
well as how answerable the questions are.

If you are ready, please click "Accept HIT" to start this task.
'''
