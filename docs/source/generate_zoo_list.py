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


def model_text(model_dict, fout):
    name = model_dict.get('title').title()
    fout.write(name)
    fout.write('\n')
    fout.write('~' * len(name))
    fout.write('\n')

    links = ''
    if 'project' in model and model['project']:
        link = model['project']
        links += '`[related project] <' + link + '/>`__ '
    if 'external_website' in model and model['external_website']:
        link = model['external_website']
        links += '`[external website] <' + link + '/>`__ '
    if links != "":
        fout.write(links + "\n")

    fout.write("\n" + model['description'])
    fout.write('\n\n')

    fout.write('Example invocation(s):\n\n')
    if 'example' in model:
        example = model['example']
    else:
        example = "python -m parlai.scripts.eval_model --model {} --task {} -mf {}".format(
            model['agent'], model['task'], model['path']
        )
    result = model.get('result', '').strip().split("\n")
    # strip leading whitespace from results
    result = [r.strip() for r in result]
    # make sure we indent for markdown though
    result = ["   " + r for r in result]
    result = "\n".join(result)
    fout.write('.. code-block:: none\n\n   {}\n   \n{}\n'.format(example, result))
    fout.write('\n')

    if 'example2' in model:
        example = model['example2']
        result = model.get('result2', '').strip().split("\n")
        # strip leading whitespace from results
        result = [r.strip() for r in result]
        # make sure we indent for markdown though
        result = ["   " + r for r in result]
        result = "\n".join(result)
        fout.write('.. code-block:: none\n\n   {}\n   \n{}\n'.format(example, result))
        fout.write('\n')


fout = open('zoo_list.inc', 'w')

for task_name in category_zoo_list:
    s = task_name.title().replace('_', ' ') + ' models\n'
    fout.write(s)
    fout.write('-' * len(s) + '\n\n')

    for model in category_zoo_list[task_name]:
        model_text(model, fout)

fout.close()
