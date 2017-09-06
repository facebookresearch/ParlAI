# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.tasks.task_list import task_list

category_order = ['QA', 'Cloze', 'Goal', 'ChitChat', 'Negotiation', 'Visual']
category_task_list = {x:[] for x in category_order}

for task_dict in task_list:
    tags = task_dict.get('tags', None)
    for tag in tags:
        if tag in category_task_list:
            category_task_list[tag].append(task_dict)

with open('task_list.inc', 'w') as fout:
    for category, task_list in category_task_list.items():
        fout.write(category + '\n')
        fout.write('-' * len(category) + '\n\n')

        for task_dict in task_list:
            id = task_dict.get('id', None)
            display_name = task_dict.get('display_name', None)
            task = task_dict.get('task', None)
            tags = task_dict.get('tags', None)
            description = task_dict.get('description', None)
            notes = task_dict.get('notes', None)

            fout.write(display_name + '\n')
            fout.write('^' * len(display_name) + '\n\n')

            fout.write('**Tag**: ``#' + id + '``\n\n')
            fout.write('**Full Path**: ``' + task + '``\n\n')
            tag_list_string = ''
            for i in range(len(tags)):
                tag_list_string += ('``#' + tags[i] + '``')
                if i < len(tags) - 1:
                    tag_list_string += ', '
            fout.write('**Group Tags**: ' + tag_list_string + '\n\n')
            fout.write('**Description**: ' + description + '\n\n')

            if notes:
                fout.write('**Notes**: ' + notes + '\n')

            fout.write('\n\n')

