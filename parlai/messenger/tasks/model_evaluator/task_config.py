# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Give a rating to a dialog between two people'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'Give a rating to a dialog between two people.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog,rating'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = \
'''\'\'\'
In this task, you are going to read a dialog between two people, and you will need to give a rating on how good the response is.<br><br>
Example:<br><br>
------------------- Task Begin ------------------- <br><br>
<b>Model Evaluator</b>:<br>
This is the author of the article . These were my picks and it 's an opinion . I did say Quantum was mediocre to bad and it 's because the trailer is so incredible and Casino Royale was so great that it was a let down . Also are you really gon na say Phantom Menace wasnt a terrible movie that had a great trailer .<br><br>
How would you rate the following response (from 0 to 10):<br><br>
True its an opinion as is my comment . I 'd say quantum of solace was meh , bland . But it had one of the best bond villains around . As for phantom menace , I 'd say it gets far more hate than it deserves . Did I personally enjoy it ? Yes . Was it a good movie ? Not especially . Did it live up to the hype ? God no ? Was it terrible ? Not even close . Attack of the clones on the other hand , that was dreck .<br><br>
<b>Worker</b>:<br>
8<br><br>
------------------- Task Done ------------------- <br><br>
If you are ready, please click "Accept HIT" to start this task.
\'\'\''''


