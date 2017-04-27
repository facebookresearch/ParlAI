# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Helper functions for defining the set of tasks in ParlAI.
   The actual task list and definitions are in the file tasks.json
"""

import copy
import json
import os 

def _preprocess(name):
    name = name.lower().replace('-', '')
    return name

def _build():
    tasks = {}
    tags = {}
    if len(tasks) > 0:
        return
    parlai_dir = (os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.realpath(__file__)))))
    task_path = parlai_dir + '/parlai/tasks/tasks.json'
    with open(task_path) as data_file:
        task_data = json.load(data_file)
    for k in task_data:
        task = _preprocess(k['id'])
        if task in tasks:
            raise RunTimeError('task ' + j + ' already exists')
        tasks[task] = [ k ]
        for j in k['tags']:
            tag = _preprocess(j)
            if tag in tasks:
                raise RuntimeError('tag ' + tag +
                                   ' is the same as a task name')
            if tag not in tags:
                tags[tag] = []
            tags[tag].append(k)            
    return tasks, tags

def _id_to_task_data(id):
    id = _preprocess(id)
    if id in tasks:
        return tasks[id]
    else:
        if id in tags:
            return tags[id]
        else:
            raise RuntimeError(id + ' task/tag not found')

 
def _id_to_task(id):
    if len(id) == 0:
        return None
    if id[0] == '#':
        data = _id_to_task_data(id[1:])
        tasks =  []
        for d in data:
            tasks.append(d['task'])
        return ','.join(tasks)
    else:
        return id


def ids_to_tasks(tasks):
    tasks = tasks.split(',')
    for k in range(len(tasks)):
       tasks[k] = _id_to_task(tasks[k])
    task = ','.join(tasks)
    return task

# Build the task list from the json file.
tasks, tags = _build()

