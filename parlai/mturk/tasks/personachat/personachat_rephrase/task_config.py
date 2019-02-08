#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search results,
and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Rehprase a Character Description'


"""A description includes detailed information about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the expanded
view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = (
    'You will rephrase some character (persona) descriptions'
)


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog,text,game'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = '''
In this task, we will show you 4 to 5 sentences with each of them describes
some person's characteristics.
<br>
Your jobs is to <b><span style="color:blue">rephrase the sentence to a new one
which is about a relative characteristic that the same person may
have.</span></b>
<br><br>
<b><span style="color:red">Do not trivially rephrase by copying the words in
the original sentences. Make a natural rephrasing.</span></b>
See examples below, BAD EXAMPLES have trivial word matching and weird
rephrasing.
<br>
<br>
<b><span style="color:blue">GOOD EXAMPLE</span></b>: rephrase "<b>My father
worked for Ford.</b>" to
"<b><span style="color:blue">My dad worked in the car industry.</span></b>"
<br>
<b><span style="color:red">BAD EXAMPLE</span></b>: rephrase "<b>My father
worked for Ford.</b>" to
"<b><span style="color:red">My dad was employed by Ford.</span></b>"
(trivial word matching like "Ford")
<br>
<br>
<b><span style="color:blue">GOOD EXAMPLE:</span></b> rephrase
"<b>I like basketball.</b>" to
"<b><span style="color:blue">I am big fan of Michael Jordan.</span></b>"
<br>
<b><span style="color:red">BAD EXAMPLE</span></b>: rephrase "<b>I like
basketball.</b>" to
"<b><span style="color:red">I am good at basketball.</span></b>"
(trivial word matching like "basketball")
<br>
<br>
<b><span style="color:blue">GOOD EXAMPLE:</span></b> rephrase "<b>I cannot
choose between lollipops and rainbows</b>" to
"<b><span style="color:blue">I like candies.</span></b>"
<br>
<b><span style="color:red">BAD EXAMPLE</span></b>: rephrase "<b>I cannot choose
between lollipops and rainbows</b>" to
"<b><span style="color:red">Suckers and brightly colored prism reflections are
two of my favorites</span></b>" (unnatural phrases like "prism reflections")
<br>
<br>
<b><span style="color:blue">GOOD EXAMPLE:</span></b> rephrase "<b>I like eating
pretzels</b>" to
"<b><span style="color:blue">I enjoy beers and beer snacks.</span></b>"
<br>
<b><span style="color:red">BAD EXAMPLE</span></b>: rephrase "<b>I like eating
pretzels</b>" to
"<b><span style="color:red">I like to chew and swallow twisted bread with
salt</span></b>" (unnatural description)
<br>
<br>
After you finish, click “Done with this HIT” to submit.
<br>
<br>
- Do not reference the task or MTurk itself when rephrasing the character.
<br>
- No racism, sexism or otherwise offensive comments, or the submission will be rejected.
'''
