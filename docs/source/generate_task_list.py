#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.tasks.task_list import task_list

category_order = ['QA', 'Cloze', 'Goal', 'ChitChat', 'Negotiation', 'Visual', 'decanlp']
category_task_list = {x: [] for x in category_order}

fout = open('task_list.inc', 'w')

s = "They consist of:  "
cnt = 0
for t in category_order:
    cnt += 1
    s += "(" + str(cnt) + ") " + t + " tasks;  "
    if cnt == len(category_order) - 1:
        s += " and "
fout.write(s[:-3] + ".\n\n")

for task_dict in task_list:
    tags = task_dict.get('tags', None)
    for tag in tags:
        if tag in category_task_list:
            category_task_list[tag].append(task_dict)

for category, tl in category_task_list.items():
    s = category + ' Tasks\n'
    fout.write(s)
    fout.write('-' * len(s) + '\n\n')

    for task_dict in tl:
        id = task_dict.get('id', None)
        display_name = task_dict.get('display_name', None)
        task = task_dict.get('task', None)
        tags = task_dict.get('tags', None)
        description = task_dict.get('description', None)
        notes = task_dict.get('notes', None)

        fout.write("**" + display_name + "**   " + description + '  ')
        # fout.write('^' * len(display_name) + '\n\n')

        urlend = task[: max(task.find(':'), len(task))]
        url = (
            "https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/"
            + urlend
        )
        # codez = "`c <" + url + ">`_"
        fout.write("[ task:`" + task + " <" + url + ">`_  tags:``#" + id + '``, ')
        tag_list_string = ''
        for i in range(len(tags)):
            tag_list_string += '``#' + tags[i] + '``'
            if i < len(tags) - 1:
                tag_list_string += ', '
        fout.write(tag_list_string + ' ]\n\n')

        # if notes:
        #     fout.write('**Notes**: ' + notes + '\n')

        fout.write('\n\n')

fout.close()
