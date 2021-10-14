#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.tasks.task_list import task_list

MAIN = "https://github.com/facebookresearch/ParlAI/tree/main"

categories = set()
for task_dict in task_list:
    categories.update(task_dict.get('tags', []))
categories = sorted(categories)
category_task_list = {x: [] for x in categories}

fout = open('task_list.inc', 'w')

s = "They consist of:  "
for t in categories:
    fout.write(f"1. {t} tasks\n")
fout.write("\n")

for task_dict in task_list:
    tags = task_dict.get('tags', [])
    if not tags:
        if 'Uncategorized' not in category_task_list:
            category_task_list['Uncategorized'] = []
        category_task_list['Uncategorized'].append(task_dict)
    for tag in tags:
        if tag in category_task_list:
            category_task_list[tag].append(task_dict)

for num_category, (category, tl) in enumerate(category_task_list.items()):
    if num_category != 0:
        fout.write("\n-----\n\n")

    fout.write(f'## {category} Tasks\n')

    for task_dict in tl:
        id = task_dict.get('id', None)
        display_name = task_dict.get('display_name', None)
        task = task_dict.get('task', None)
        tags = task_dict.get('tags', None)
        description = task_dict.get('description', None)
        notes = task_dict.get('notes', None)
        code_urlend = task[: max(task.find(':'), len(task))]
        code_url = f"{MAIN}/parlai/tasks/{code_urlend}"
        links = task_dict.get("links", {})
        assert isinstance(links, dict), f"task {id} is poorly formatted"
        urls = [(k, v) for k, v in links.items()]
        urls.append(("code", code_url))

        urls_md = ", ".join(f"[{k}]({v})" for k, v in urls)
        fout.write(f"### {display_name.title().replace('_', ' ')}\n")
        fout.write(f"_Usage_:  `--task {task}`\n\n")
        fout.write(f"_Links_:  {urls_md}\n\n")
        if description:
            fout.write(description + "\n")
        if notes:
            fout.write(":::{admonition,note} Notes\n")
            fout.write(notes + "\n")
            fout.write(":::\n")
        fout.write("\n\n")

fout.close()
