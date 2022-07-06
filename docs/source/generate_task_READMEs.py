#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.tasks.task_list import task_list
import os

tasks = {}

for task_dict in task_list:
    id = task_dict.get('id', None)
    display_name = task_dict.get('display_name', None)
    task_detailed = task_dict.get('task', None)
    if ':' in task_detailed:
        task = task_detailed[0 : task_detailed.find(':')]
    else:
        task = task_detailed
    tags = task_dict.get('tags', None)
    description = task_dict.get('description', None)
    notes = task_dict.get('notes', None)

    str = "Task: " + display_name + '\n'
    str += '=' * len(str)
    description = description.replace('Link: http', '\n\nLink: http')
    str += "\nDescription: " + description + "\n\n"
    str += "Tags: #" + id + ", "
    tag_list_string = ''
    for i in range(len(tags)):
        tag_list_string += '#' + tags[i] + ''
        if i < len(tags) - 1:
            tag_list_string += ', '
    str += tag_list_string + '\n'
    if notes:
        str += "\nNotes: " + notes + "\n"
    str += "\n"
    if task not in tasks:
        tasks[task] = str
    else:
        tasks[task] += '\n\n' + str

for t in tasks.keys():
    path = os.path.join("../../parlai/tasks/", t, 'README.md')
    fout = open(path, 'w')
    fout.write(tasks[t])
    fout.close()
