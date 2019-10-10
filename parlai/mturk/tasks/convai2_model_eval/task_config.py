#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Talk with our Chat-bot!'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config[
    'hit_description'
] = 'You will chat to a chat-bot.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config[
    'task_description'
] = """
<br>
<b><h4>Task Description</h4></b>
<br>
(You can keep accepting new HITs after you finish your current one, so keep working on it if you like the task!)
<br>
<b>
In this task you will chitchat with a chat-bot with is a deep learning model.</b>
The dialogue model is pre-trained with large-scale news data.
<br>
Chat with the other user naturally.
<br>
Try to cut into the topic, no need to be polite (e.g. no need to begin with "hello").
<br>
<b>Send short messages, <span style="color:red">max 20 words</span>.</b>
<br>
After a given number of turns, you will be asked to <b>briefly</b> rate the bot on metrics like <b>fluency, engagingness, and consistency</b>.
<br>
There is a <b>2 min</b> time limit for each turn.
<br>
<br>
- Do not reference the task or MTurk itself during the conversation.
<br>
<b><span style="color:red">- No racism, sexism or otherwise offensive comments, or the submission will be rejected and we will report to Amazon.</b></span>
<br>
<br>
"""
