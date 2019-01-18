# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Respond to Comment on an Image'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'You will write an engaging response '
'to a comment on an image.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'comment'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = \
    '''
<h2><b>Description</b></h2>
In this task, you will be shown 5 images, each of which has a comment about the image.
The goal of this task is to write an engaging response to this comment
as if you were continuing a dialog about the image.
<br>
<br>
<h4><b>STEP 1</b></h4> With each new photo, you will be given a
<b>personality trait</b> that you will try to emulate in your
response to the comment on the image.
For example, you might be given "<b>snarky</b>" or "<b>sentimental</b>".
The personality describes
<em><b>YOU</b></em>, not the picture. It is <em>you</em> who is snarky or sentimental,
not the contents of the image nor the original comment about the image.
<br>
<br>
<h4><b>STEP 2</b></h4> You will then be shown an image and a comment that goes with
the image, for which you will write a response
<em>in the context of your given personality trait</em>.
Please make sure your response has at least <b>three words</b>. Note that these are
<em>responses</em> to the comments on the image, and not simply image captions.
<br>
<br>
E.g., you may be shown an image of a tree, and a comment that says
"I'd love to climb that tree!." If your personality trait is "<b>snarky</b>,"
you might say "I bet you couldn't if you tried!", or if you were "<b>sentimental</b>"
you might say "I remember when I used to climb trees back in the good old days..."
<br>
<br>
NOTE: you will receive a new personality for each new image. Please do not simply
copy the personality into your response, and
<b>please try not to use the text in the image when forming a comment</b>.
<br>
<br>
<h4><b>REWARD/BONUS</b></h4>
To complete this task, <b><span style="color:blue">you must write a response to
comments on ALL 5 images.</span></b>
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
<span style="color:blue"><b>1. Be aware the response you enter will be made public,
so write as you would e.g. on a public social network like Twitter.</b></span>
<br>
2. Please do not reference the task or MTurk itself in the response. Additionally,
<b>please try not to use the text in the image when forming a response</b>.
<br>
3. We will reject HITs that do not display any sense that you have looked at the
image or the comment while forming your response. That is, if the comment has
nothing to do with neither the image nor the comment, we will not accept it.
<br>
4. Likewise, we will reject HITs that do not display any sense that you have
looked at the personality while forming the response.
<br>
5. Please do not write a response that involves any level of discrimination,
racism, sexism and offensive religious/politics comments,
otherwise the submission will be rejected.
<br>
<br>
<br>
If you are ready, please click "Accept HIT" to start this task.
'''
