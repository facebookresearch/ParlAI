#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}

end_info = """
<h4><span style="color:blue"><b>Reward/Bonus</b></span></h4>
If you complete the task, you will receive $1.62.
<br>
<b>We will reward engaging and knowledgeable chats with a bonus.</b>
<br>
<br>
<h4><span style="color:blue"><b>Close Window/Timeout/Return HIT</b></span></h4>
Once the conversation has started, close window/timeout or return HIT during the
chat will result in
<b>HIT EXPIRED</b> to you and NO reward paid.
<br>
<br>
<h4><span style="color:blue"><b>Important Notice</b></span></h4>
1. <b>Be aware the conversations you have will be made public, so act as you
would e.g. on a public social network like Twitter.</b>
<br>
2. Please do not send long messages: messages cannot exceed 30 words.
<br>
3. Please do not reference the task or MTurk itself during the conversation,
but speak naturally to the other person.
<br>
4. Please do not send any message that could make others uncomfortable,
including any level of discrimination, racism, sexism and offensive
religious/politics comments, otherwise the submission will be rejected.
<br>
<br>
<br>
"""

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Chat with a Real Person'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'You will chat with another person, either freely '
'or in the context of information provided for each response.'


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
<h2><b>Description</b></h2>
In this task, you will have a conversation with another person. The goal of
this task is to go into depth about something that interests you or the other
player, while keeping the conversation engaging and fun.
<br>
<br>
<h4><span style='color:blue'><b>Sample Conversation</b></span></h4>
<b>Person 1</b>: Hi! I really like board games.
<br>
<b>Person 2</b>: Oo, what type of board games?
<br>
<b>Person 1</b>: I like strategy games, especially ones that are sci-fi
<br>
<b>Person 2</b>: I love Risk, but it takes place on earth, so not sci-fi,
and it takes forever
<br>
<b>Person 1</b>: Right? How do you feel about cards against humanity?
<br>
<b>Person 2</b>: Cards against humanity is fun but a little too risque for me
<br>
<br>
{}
If you are ready, please click "Accept HIT" to start this task.
'''.format(
    end_info
)

task_config[
    'wizard_onboarding'
] = '''
<h2>You have just met the other person, who seems quite curious, and you are
eager to discuss a topic with them!</h2>
<br>
You will try to inform your conversation partner about a topic that one of you
will choose. After a topic is chosen, you will receive information about that
topic that will be visible throughout the chat.
<br>
Additionally, after any message is sent in the chat, you will have access
to a set of relevant information.
<br>
Try to use one of these sentences to answer your partner's questions, and in
general have an engaging conversation.
<br>
<br>
<b><span>Please indicate which sentence you used by checking the box next to it.
If you do not use any sentence, check the "No Sentence Used" box.</span></b>
If you use a sentence even slightly, please check it.
<b>If you do not use enough external
information throughout the conversation, you may be prevented from
doing more of these HITs</b>.
<br>
<br>
Some Guidelines:
<br><br>
<ol>
<li>Please do not simply copy and paste the provided checked sentence
- there's no fun in that!</li>
<li>Do not use “know-it-all” phrases such as "did you know" in your responses
- e.g., the response "did you know that the Berlin Wall was demolished in 1989"
will not be accepted — be fun and engaging!</li>
<li><b>Important Note</b>: if you do not use enough external information in
your responses throughout the conversation, you may be prevented from
completing this task in the future.</li>
</ol>
<br>
After a minimum number of turns, you will be able to click the
DONE button to finish the chat.
To guarantee an efficient conversation, there is a time limit for sending a
message to another person (3 mins).
<b>Note: we will reward engaging and knowledgeable chats with BONUSES.</b>
<br>
<br>
<h4><span style="color:blue"><b>Conversation Outline</b></span></h4>
Thus, a conversation will proceed as follows (from your perspective):<br>
<ol>
<li>You or your partner will pick an initial conversation topic from the list
provided, at which point you will receive information about the topic, and
then the conversation will begin.</li>
<li>When your partner sends you a message, you will look at the relevant
information, and check a sentence you will use to construct a response.
If you do not use any sentences, you must check the "No Sentence Used" option</li>
<li>You will respond to your partner, basing your response on the chosen sentence.</li>
</ol>
<br>
And the conversation repeats until the minimum number of turns is reached,
at which point your partner will evaluate your level of engagingness.
<br>
<br>
<h4><span style="color:blue"><b>Sample Conversation</b></span></h4>
<b>Them</b>: Hi! I really like board games.
<br>
<b>You</b>: Oo, what type of board games?
<br>
<b>Them</b>: I like strategy games, especially ones that are sci-fi
<br>
<b>You</b>: I love Risk, but it takes place on earth, so not sci-fi, and it
takes forever
<br>
<b>Them</b>: Right? How do you feel about cards against humanity?
<br>
<b>You</b>: Cards against humanity is fun but a little too risque for me
<br>
<br>
{}
<br>
<br>
'''.format(
    end_info
)
task_config[
    'apprentice_onboarding'
] = '''
<h2> You have just met the other person, who seems quite knowledgable, and you are
curious to discuss a topic with them!</h2>
<br>
You will try to learn from your conversation partner about a topic that one of you will
choose. Feel free to dive deep on specifics - your partner will have access to
external information that will help them craft their response.
<br>
<br>
After a minimum number of turns, you will be able to click the DONE button
to finish the chat.
To guarantee an efficient conversation, there is a time limit for sending a message
to another person (2 mins).
<br>
<br>
<h4><span style="color:blue"><b>Conversation Outline</b></span></h4>
Thus, a conversation will proceed as follows: (from your perspective)<br>
<ol>
<li>You or your partner will pick an initial conversation topic from the
list provided, and open the conversation.</li>
<li>You will try to learn about the chosen topic</li>
<li>After you send a message, your partner will respond, continuing the
conversation and hopefully providing more information about the topic.</li>
</ol>
<br>
And the conversation continues until the minimum number of chat turns is
reached, at which point you will evaluate the quality of your conversation
partner, based on how relevant, engaging, and <b>knowledgeable</b> they were.
<br>
<br>
<h4><span style="color:blue"><b>Sample Conversation</b></span></h4>
<br>
<br>
<b>You</b>: Hi! I really like board games.
<br>
<b>Them</b>: Oo, what type of board games?
<br>
<b>You</b>: I like strategy games, especially ones that are sci-fi
<br>
<b>Them</b>: I love Risk, but it takes place on earth, so not sci-fi,
and it takes forever
<br>
<b>You</b>: Right? How do you feel about cards against humanity?
<br>
<b>Them</b>: Cards against humanity is fun but a little too risque for me
<br>
<br>
{}
'''.format(
    end_info
)
