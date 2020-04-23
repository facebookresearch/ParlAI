#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generalized and miscellaneous teachers.
"""

import copy
import random
from typing import List

from parlai.core.agents import Agent, create_agent_from_shared
from parlai.core.metrics import aggregate_named_reports
from parlai.core.opt import Opt
from parlai.core.teachers import create_task_agent_from_taskname, Teacher


class LabelToTextTeacher(Teacher):
    """
    Teacher that will shift message['labels'][0] into message['text'] for whatever task
    is specified with --label-to-text-task.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        # Teacher has no args, and so we don't add them
        parser = argparser.add_argument_group('LabelToText args')
        parser.add_argument(
            '-l2t-task' '--label-to-text-task',
            type=str,
            help='The task whose labels will get shifted into the text field',
        )

    def __init__(self, opt: Opt, shared=None):
        if ',' in opt['task']:
            raise ValueError('LabelToTextTeacher cannot be used with multiple tasks!')
        self.id = opt['task']
        if shared and 'task' in shared:
            # TODO: revise from here
            self.task = create_agent_from_shared(shared['task'])
        else:
            tasks = opt['task'].split(',')
            for k in tasks:
                k = k.strip()
                if k:
                    opt_singletask = copy.deepcopy(opt)
                    opt_singletask['task'] = k
                    self.tasks = create_task_agent_from_taskname(opt_singletask)
        self.task_idx = -1
        self.new_task = True
        self.random = opt.get('datatype') == 'train'
        # Make multi-task task probabilities.
        self.cum_task_weights = [1] * len(self.tasks)
        self.task_choices = range(len(self.tasks))
        weights = self.opt.get('multitask_weights', [1])
        sum = 0
        for i in self.task_choices:
            if len(weights) > i:
                weight = weights[i]
            else:
                weight = 1
            self.cum_task_weights[i] = weight + sum
            sum += weight

    def num_examples(self):
        """
        Return the number of examples.
        """
        if not hasattr(self, 'num_exs'):
            # num_examples is sum of all examples in all tasks
            tasks_num_exs = [t.num_examples() for t in self.tasks]
            if any(num is None for num in tasks_num_exs):
                self.num_exs = None
            else:
                self.num_exs = sum(tasks_num_exs)
        return self.num_exs

    def num_episodes(self):
        """
        Return the number of episodes.
        """
        if not hasattr(self, 'num_eps'):
            # num_episodes is sum of all num_episodes in all tasks
            tasks_num_eps = [t.num_episodes() for t in self.tasks]
            if any(num is None for num in tasks_num_eps):
                self.num_eps = None
            else:
                self.num_eps = sum(tasks_num_eps)
        return self.num_eps

    def observe(self, observation):
        """
        Make an observation.
        """
        return self.tasks[self.task_idx].observe(observation)

    def act(self):
        """
        Act on the previous observation.
        """
        if self.new_task:
            self.new_task = False
            if self.random:
                # select random teacher
                self.task_idx = random.choices(
                    self.task_choices, cum_weights=self.cum_task_weights
                )[0]
            else:
                # do at most one full loop looking for unfinished task
                for _ in range(len(self.tasks)):
                    self.task_idx = (self.task_idx + 1) % len(self.tasks)
                    if not self.tasks[self.task_idx].epoch_done():
                        # if this task has examples ready, break
                        break
                if self.tasks[self.task_idx].epoch_done():
                    # all tasks are done, so return empty action table
                    return {'episode_done': True}
        t = self.tasks[self.task_idx].act()
        if t['episode_done']:
            self.new_task = True
        return t

    def epoch_done(self):
        """
        Return whether all subtasks are completed.
        """
        for t in self.tasks:
            if not t.epoch_done():
                return False
        return True

    # return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        """
        Report aggregated metrics across all subtasks.
        """
        return aggregate_named_reports(
            {t.getID(): t.report() for t in self.tasks},
            micro_average=self.opt.get('aggregate_micro', False),
        )

    def reset(self):
        """
        Reset all subtasks.
        """
        for t in self.tasks:
            t.reset()

    def reset_metrics(self):
        """
        Reset metrics for each subtask.
        """
        for t in self.tasks:
            t.reset_metrics()

    def save(self):
        """
        Save the subtask.
        """
        self.task.save()

    def share(self):
        """
        Share the subtask.
        """
        shared = {}
        shared['task'] = self.task.share()
        return shared

    def shutdown(self):
        """
        Shutdown the subagent.
        """
        self.task.shutdown()

    def update_counters(self):
        self.task.update_counters()
