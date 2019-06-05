# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.tasks.task_list import task_list

fout = open('/tmp/t', 'w')

for task_dict in task_list:
    id = task_dict.get('id', None)
    display_name = task_dict.get('display_name', None)
    task = task_dict.get('task', None)
    tags = task_dict.get('tags', None)
    description = task_dict.get('description', None)
    notes = task_dict.get('notes', None)

    fout.write("Task: " + display_name + "\nDescription: " + description + "\n")
    # fout.write('^' * len(display_name) + '\n\n')

    pathend = task[: max(task.find(':'), len(task))]
    path = "../../parlai/tasks/" + pathend
    fout.write(path)
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
