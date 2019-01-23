#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Create a Character'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = 'You will create a character (persona) by several sentences.'


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog,text,game'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = '''
In this task, you will be asked to create a character (persona) description using <b><span style="color:blue">5 sentences</span></b>. An example would be:
<br>
<br>
"I am a vegetarian. I like swimming.  My father used to work for Ford.  My favorite band is Maroon5. I got a new job last month, which is about advertising design."
<br>
<br>
Please make each sentence short, max 15 words per sentence.
<br>
<b><span style="color:blue">Please do not use sensitive personal information in creating the character, as it may be publicly released.</span></b>
<br>
<br>
After you finish, click “Done with this HIT” to submit.
<br>
<br>
- Do not reference the task or MTurk itself when creating the character.
<br>
- No racism, sexism or otherwise offensive comments, or the submission will be rejected.
'''
