# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Choose Most Engaging Comment on an Image'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'You will choose the most engaging '
'comment on an image.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'image'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = \
    '''
<h2><b>Description</b></h2>
In this task, you will be shown 5 images, each with two comments.
The goal of this task is to pick which comment is the <b><span style="color:blue">
most engaging (interesting, captivating, attention-grabbing)</span></b>.
<br>
<br>
<h4><b>STEP 1</b></h4> You will be shown an image and two comments.
Additionally, you may be shown the personality of the person who wrote the image.
<br>
<br>
E.g., you may be shown an image of a tree, and the following two comments:
<br> 1. "A tree in a park in the summer time"
<br> 2. "What an absolutely beautiful tree! I would put this in my living room
it's so extravagent!"
<br>
<br>
And, you may be shown a personality, e.g. 'Cheerful'.
<br>
<br>
<h4><b>STEP 2</b></h4> You will choose which comment is <b><span
style="color:blue">more engaging</span></b>.
<br>
<br>
E.g. in the example above, the second comment is more engaging than the first.
<br>
<br>
<h4><b>STEP 3</b></h4> You will describe why you feel the comment you chose is
<b><span style="color:blue">more engaging</span></b>.
<br>
<br>
You will then write a reason indicating why you believe this comment is
the more engaging one.
<br>
E.g. in the example above, you may say "The comment I chose was more lively
and entertaining."
<br>
<h4><b>REWARD/BONUS</b></h4>
To complete this task, <b><span style="color:blue">you must rank the comments
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
