# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Helper functions for defining the set of tasks in ParlAI.
The actual task list and definitions are in the file task_list.py
"""
from .task_list import task_list
from collections import defaultdict
import copy
import json
import os

def _preprocess(name):
    return name.lower().replace('-', '')

def _build(task_list):
    tasks = {}
    tags = defaultdict(list)

    for t in task_list:
        task = _preprocess(t['id'])
        tasks[task] = [t]
        for j in t['tags']:
            tag = _preprocess(j)
            if tag in tasks:
                raise RuntimeError('tag ' + tag +
                                   ' is the same as a task name')
            tags[tag].append(t)
    return tasks, tags

def _id_to_task_data(t_id):
    t_id = _preprocess(t_id)
    if t_id in tasks:
        # return the task assoicated with this task id
        return tasks[t_id]
    elif t_id in tags:
        # return the list of tasks for this tag
        return tags[t_id]
    else:
        # should already be in task form
        raise RuntimeError('could not find tag/task id')


def _id_to_task(t_id):
    if t_id[0] == '#':
        # this is a tag, so return all the tasks for this tag
        return ','.join((d['task'] for d in _id_to_task_data(t_id[1:])))
    else:
        # this should already be in task form
        return t_id


def ids_to_tasks(ids):
    if ids is None:
        raise RuntimeError('No task specified. Please select a task with ' +
                           '--task {task_name}.')
    return ','.join((_id_to_task(i) for i in ids.split(',') if len(i) > 0))

# Build the task list from the json file.
tasks, tags = _build(task_list)
