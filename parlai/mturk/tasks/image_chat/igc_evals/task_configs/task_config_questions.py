#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Rate Quality of Questions in Context of Image'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'You will rate the quality of questions in '
'the context of a discussion about an image.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'image,rate'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config[
    'task_description'
] = '''
<h2><b>Description</b></h2>
In this task, you will be shown 5 images. For each image, there will be
a contextual statement and a selection of various questions regarding
the image and in response to the statement.
The goal of this task is to rate the quality of these questions.
<br>
<br>
<h4><b>STEP 1</b></h4> You will be shown an image, some textual context,
and a set of candidate questions in response to the textual context.
<br>
<br>
E.g., you may be shown an image of a tree; some textual context, i.e.
"An amazing tree for climbing."; and, some candidate questions:
<br> 1. "Do you think you could really climb that tree?"
<br> 2. "Is it time for dinner yet?"
<br>
<h4><b>STEP 2</b></h4> You will rate each candidate question on a scale from 1 to 3,
where 3 is the <b>highest</b> quality and 1 is the <b>lowest</b> quality.
<br>
<br>
E.g. in the example above, you might give the first question a "3" rating and
the second question a "1" rating.
<br>
<br>
<h4><b>REWARD/BONUS</b></h4>
To complete this task, <b><span style="color:blue">you must rate the questions
on ALL 5 images.</span></b>
If you complete the task, you will receive $0.46.
<br>
<br>
<br>
<h4><b>CLOSE WINDOW/TIMEOUT/RETURN HIT</b></h4>
Once the task has started, close window/timeout or return HIT will result in
<b><span style="color:blue">HIT EXPIRED</span></b> to you and NO reward paid.
<br>
<br>
<br>
If you are ready, please click "Accept HIT" to start this task.
'''
