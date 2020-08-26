#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.zoo.model_list import model_list

category_zoo_list = {}
for model_dict in model_list:
    task = model_dict.get('task', None)
    if ':' in task:
        task = task[0 : task.find(':')]  # strip detailed task name
    if task not in category_zoo_list:
        category_zoo_list[task] = []
    category_zoo_list[task].append(model_dict)


def example_to_code(example, result):
    if not result:
        return f'```none\nexample\n```'
    result = result.strip().split("\n")
    # strip leading whitespace from results
    result = [r.strip() for r in result]
    # make sure we indent for markdown though
    result = "\n".join(result)
    return f'```none\n{example}\n\n{result}\n```'


def model_text(model_dict, fout):
    name = model_dict.get('title').title()
    fout.write(f'### {name}')
    fout.write('\n')

    links = ''
    if 'project' in model and model['project']:
        link = model['project']
        links += f'[[related project]]({link})'
    if 'external_website' in model and model['external_website']:
        link = model['external_website']
        links += f'[[external website]]({link})'
    if links != "":
        fout.write(links + "\n")

    fout.write("\n" + model['description'])
    fout.write('\n\n')

    fout.write('Example invocation(s):\n\n')
    if 'example' in model:
        example = model['example']
    else:
        example = "parlai eval_model --model {} --task {} -mf {}".format(
            model['agent'], model['task'], model['path']
        )
    fout.write(example_to_code(example, model.get('result')))
    fout.write('\n\n')

    if 'example2' in model:
        fout.write(example_to_code(model['example2'], model.get('result2')))
        fout.write('\n\n')


fout = open('zoo_list.inc', 'w')

for task_name in category_zoo_list:
    s = task_name.replace('_', ' ')
    if s[0] == s[0].lower():
        s = s.title()
    fout.write(f'## {s} models\n')

    for model in category_zoo_list[task_name]:
        model_text(model, fout)

    fout.write('\n\n---------\n\n')

fout.close()
