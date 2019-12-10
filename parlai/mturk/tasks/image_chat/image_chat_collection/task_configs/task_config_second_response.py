#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Continue a Dialog on an Image'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'You will continue a dialog on an image.'


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
In this task, you will imagine that you are speaking with your
friend about 5 separate images.
For each image, you will be shown "your" initial comment on the image, and your friend's
response to the comment.
The goal of this task is to write an engaging response to your friend as if you were
continuing a dialog about the image.
<br>
<br>
<h4><b>STEP 1</b></h4> With each new photo, you will be given a <b>personality trait</b>
that you will try to emulate in your response.
For example, you might be given "<b>adventurous</b>". The personality describes
<em><b>YOU</b></em>, not the picture. It is <em>you</em> who is adventurous,
not the contents of the image.
<br>
<br>
<h4><b>STEP 2</b></h4> You will then be shown an image, "your" initial comment
that goes with the image, and your friend's response.
You will continue the dialog by responding to your friend's response
<em>in the context of your given personality trait</em>.
Please make sure your response has at least <b>three words</b>.
Note that these are not simply image captions,
but <b>engaging</b> responses.
<br>
<br>
E.g., you may be shown an image of a tree, and you may be given the
personality "<b>adventurous</b>." Then, under the image, you might see:
<br>
<br>
1. <b>Your Comment</b>: "I'd love to climb that tree!"
<br>2. <b>Your Friend's Response</b> "I bet you couldn't if you tried!"
<br>
<br>
You might respond, "I love the thrill of climbing, so I will anyway!"
<br>
<br>
NOTE: you will receive a new personality for each new image.
Please do not simply copy the personality into your response, and
<b>please try not to use the text in the image when forming a comment</b>.
<br>
<br>
<h4><b>REWARD/BONUS</b></h4>
To complete this task, <b><span style="color:blue">you must
write a response on ALL 5 images.</span></b>
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
image or the comment while forming your response.
That is, if the comment has nothing to do with neither the image nor the comment,
we will not accept it.
<br>
4. Likewise, we will reject HITs that do not display any sense that you have
looked at the personality while forming the response.
<br>
5. Please do not write a response that involves any level of discrimination,
racism, sexism and offensive religious/politics comments,
otherwise the submission will be rejected.
<br>
6. These HITs are approved within 5 days of completion.
<br>
<br>
<br>
If you are ready, please click "Accept HIT" to start this task.
'''
