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
task_config['hit_title'] = 'Play a character and chat! [You can keep accepting new HITs]'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'You will first need to enter several sentences to create a persona, and then chat with another person.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = \
'''
<h4><b>DESCRIPTION</b></h4>
The goal of this task is to have a dialogue from the perspective of a “persona”: you will chat with another MTurker, while both of you adopt a different role or character.
<br>
<br>
<b>STEP 1</b>: you will be asked to create your persona in a few sentences.
<b><span style="color:blue">Please do not use sensitive personal information in creating your persona, as the personas may be publicly released.</span></b> For example, you can create something like:
<br>
<br>
"I am a vegetarian. I like swimming. My father used to work for Ford. My favorite band is Maroon5. I got a new job last month, which is about advertising design."
<br>
<br>
<b>STEP 2</b>: You will then be connected to another MTurker and have a free conversation with her/him.
The goal of the chat is to get to know each other.
<b><span style="color:blue">Please involve and stick to your persona during the chat, i.e. speak to the other person AS IF YOU ARE THE PERSONA YOU CREATED.</span></b>
<br>
<br>
After a minimum number of turns, you will be able to click the [DONE] button to finish the chat.
To guarantee an efficient conversation, there are time limits both for creating the persona (time limit 4 mins) and for sending a message to another person (time limit 2 mins).
<br>
<br>
<br>
<h4><b>REWARD/BONUS</b></h4>
If you successfully finished STEP 1 (describing your persona) and started STEP 2, you will get $0.20 reward.
<br>
You will get $0.03 bonus for each chat turn (up to 10 turns).
<br>
<br>
<br>
<h4><b>CLOSE WINDOW/TIMEOUT/RETURN HIT</b></h4>
Once the conversation has started, close window/timeout or return HIT during the chat will result in
<b><span style="color:blue">HIT EXPIRED</span></b> to you and NO bonus/reward paid, while the other person will get accepted.
<br>
<br>
<br>
<h4><b>IMPORTANT NOTICE</b></h4>
<span style="color:blue"><b>1. Be aware the persona/conversations you enter will be made public, so act as you would e.g. on a public social network like Twitter.</b></span>
<br>
2. Please do not send long messages: messages cannot exceed 30 words.
<br>
3. Please do not reference the task or MTurk itself during the conversation, but speak naturally to the other person.
<br>
4. Please do not send any message that could make others uncomfortable, including any level of discrimination, racism, sexism and offensive religious/politics comments, otherwise the submission will be rejected.
<br>
<br>
<br>
If you are ready, please click "Accept HIT" to start this task.
'''
