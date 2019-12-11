#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Comment on an Image'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'You will write an engaging comment for an image.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'comment'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config[
    'task_description'
] = '''
<h2><b>Description</b></h2>
In this task, you will be shown 5 images, and will write a comment about each image.
The goal of this task is to write something about an image that someone else would
find engaging.
<br>
<br>
<h4><b>STEP 1</b></h4> With each new photo, you will be given a <b>personality trait</b>
that you will try to emulate in your comment. For example, you might be given
"<b>snarky</b>" or "<b>glamorous</b>". The personality describes
<em><b>YOU</b></em>, not the picture. It is <em>you</em> who is snarky or
glamorous, not the contents of the image.
<br>
<br>
<h4><b>STEP 2</b></h4> You will then be shown an image, for which you will write
a comment <em>in the context of your given personality trait</em>.
Please make sure your comment has at least <b>three words</b>. Note that these are
<em>comments</em>, not captions.
<br>
<br>
E.g., you may be shown an image of a tree. If you are "<b>snarky</b>",
you might write "What a boring tree, I bet it has bad wood;" or,
if you were "<b>glamorous</b>", you might write "What an absolutely beautiful tree!
I would put this in my living room it's so extravagent!"
<br>
<br>
NOTE: you will receive a new personality for each new image. Please do not
simply copy the personality into your comment, and <b>please try not to use
the text in the image when forming a comment</b>.
<br>
<br>
<h4><b>REWARD/BONUS</b></h4>
To complete this task, <b><span style="color:blue">you must comment on ALL 5 images.
</span></b>
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
<h4><b>IMPORTANT NOTICE</b></h4>
<span style="color:blue"><b>1. Be aware the comment you enter will be made public,
so write as you would e.g. on a public social network like Twitter.</b></span>
<br>
2. Please do not reference the task or MTurk itself in the comment. Additionally,
<b>please try not to use the text in the image when forming a comment</b>.
<br>
3. We will reject HITs that do not display any sense that you have looked at the
image while forming the comment. That is, if the comment has nothing to do with the
image, we will not accept it.
<br>
4. Likewise, we will reject HITs that do not display any sense that you have
looked at the personality while forming the comment.
<br>
5. Please do not comment anything that involves any level of discrimination,
racism, sexism and offensive religious/politics comments, otherwise
the submission will be rejected.
<br>
<br>
<br>
If you are ready, please click "Accept HIT" to start this task.
'''
