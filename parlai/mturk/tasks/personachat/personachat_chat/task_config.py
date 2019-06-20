#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Play a character and chat!'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config[
    'hit_description'
] = 'You will chat to another person while adopting a specific persona.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config[
    'task_description'
] = '''
<br>
<b><h4>Task Description</h4></b>
<br>
(You can keep accepting new HITs after you finish your current one, so keep working on it if you like the task!)
<br>
<b>In this task you will chitchat with another worker, playing the part of a given character.</b>
For example, your given character could be: <br><br> I am a vegetarian. I like swimming. My father used to work for Ford. My favorite band is Maroon5. I got a new job last month, which is about advertising design.
<br>
<br>
Chat with the other person naturally and <b><span style="color:blue">try to get to know each other, i.e.
both ask questions and answer questions of your chat partner
at the same time sticking to your own characters<span style="color:blue"></b>.
<br>
<br>
<b><span style="color:blue">You will get bonus for high quality dialogs.</span></b>
<b>Send short messages, <span style="color:red">max 15 words</span>.</b>
<b>Do not trivially copy the character descriptions into the message.</b>
After a given number of turns, click “DONE" to finish the chat.
There is a <b>2 min</b> time limit for each turn.
<br>
<br>
- Do not reference the task or MTurk itself during the conversation.
<br>
<b><span style="color:red">- No racism, sexism or otherwise offensive comments, or the submission will be rejected and we will report to Amazon.</b></span>
'''
