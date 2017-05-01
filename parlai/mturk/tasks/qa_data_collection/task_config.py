# Copyright 2004-present Facebook. All Rights Reserved.
# Task config for MTurk task

task_config = {}

# MTurk config related
task_config['hit_title'] = 'Ask and answer a question about a paragraph'
task_config['hit_description'] = 'Ask and answer a question about a paragraph.'
task_config['hit_keywords'] = 'chat,question,answer'
task_config['hit_reward'] = 0.05

# Task specific
task_config['teacher_agent_id'] = 'teacher'
task_config['worker_agent_id'] = task_config['teacher_agent_id']

# Required for all tasks
# Task description shown on the left side of the HIT chat page
task_config['task_description'] = \
'''\'\'\'
In this task, you will need to ask a question about a paragraph, and then provide your own answer to it.<br><br>
Example:<br><br>
------------------- Task Begin ------------------- <br><br>
<b>QA Collector</b>:<br>
New Haven\'s greatest culinary claim to fame may be its pizza, which has been claimed to be among the best in the country, or even in the world. New Haven-style pizza, called "apizza" (pronounced ah-BEETS, [a'pitts] in the original Italian dialect), made its debut at the iconic Frank Pepe Pizzeria Napoletana (known as Pepe\'s) in 1925. Apizza is baked in coal- or wood-fired brick ovens, and is notable for its thin crust. Apizza may be red (with a tomato-based sauce) or white (with a sauce of garlic and olive oil), and pies ordered "plain" are made without the otherwise customary mozzarella cheese (originally smoked mozzarella, known as "scamorza" in Italian). A white clam pie is a well-known specialty of the restaurants on Wooster Street in the Little Italy section of New Haven, including Pepe\'s and Sally\'s Apizza (which opened in 1938). Modern Apizza on State Street, which opened in 1934, is also well-known.<br><br>Please provide a question given this context.<br><br>
<b>Worker</b>:<br>
What is apizza baked in?<br><br>
<b>QA Collector</b>:<br>
Thanks. And what is the answer to your question?<br><br>
<b>Worker</b>:<br>
It's baked in coal- or wood-fired brick ovens.<br><br>
------------------- Task Done ------------------- <br><br>
If you are ready, please click "Accept HIT" to start this task.
\'\'\''''


