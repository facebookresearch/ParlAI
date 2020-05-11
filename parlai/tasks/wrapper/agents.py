#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Teachers that wrap around other teachers, for instance, to modify message fields while
keeping the examples/episodes the same.

This is useful when working with agents that expect examples to be in a certain format,
for instance a classifier that classifies the "text" field of a message. The meta-
teachers in this module can be used to avoid writing several different nearly identical
variants of different teachers: for instance, if you want to flatten examples and strip
away all but the previous utterance in the 'text' field for several different teachers,
it would be much easier to do so with one teacher in this module than with a brand new
teacher for each of the original teachers.
"""

import copy

from parlai.core.agents import create_agent_from_shared
from parlai.core.opt import Opt
from parlai.core.teachers import create_task_agent_from_taskname, Teacher


class LabelToTextTeacher(Teacher):
    """
    Teacher that will shift message['labels'][0] into message['text'] for whatever task
    is specified with --label-to-text-task.

    Because the dialogue history is effectively overwritten by this action, all episodes
    will be flattened into one example each.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        # Teacher has no args
        parser = argparser.add_argument_group('LabelToText args')
        parser.add_argument(
            '-l2tt',
            '--label-to-text-task',
            type=str,
            help='The task whose labels will get shifted into the text field',
        )

    def __init__(self, opt: Opt, shared=None):
        if ',' in opt['task']:
            raise ValueError('LabelToTextTeacher cannot be used with multiple tasks!')
        self.id = opt['task']
        self.opt = opt
        if shared and 'task' in shared:
            self.task = create_agent_from_shared(shared['task'])
        else:
            opt_singletask = copy.deepcopy(opt)
            opt_singletask['task'] = opt['label_to_text_task']
            self.task = create_task_agent_from_taskname(opt_singletask)[0]

    def num_examples(self):
        """
        Return the number of examples.
        """
        return self.task.num_examples()

    def num_episodes(self):
        """
        Return the number of episodes.

        Because the dataset is flattened, there will be one episode per example.
        """
        return self.task.num_examples()

    def observe(self, observation):
        """
        Make an observation.
        """
        return self.task.observe(observation)

    def act(self):
        """
        Act on the previous observation.
        """
        act = self.task.act()
        new_act = copy.deepcopy(act)
        if 'labels' in act:
            labels = act['labels']
            assert len(labels) == 1
            new_act.force_set('text', labels[0])
            new_act.force_set('labels', [''])
        elif 'eval_labels' in act:
            labels = act['eval_labels']
            assert len(labels) == 1
            new_act.force_set('text', labels[0])
            new_act.force_set('eval_labels', [''])
        else:
            assert 'text' not in act and act['episode_done'] is True
        new_act.force_set('episode_done', True)  # Clear the dialogue history
        return new_act

    def epoch_done(self):
        """
        Return whether the subtask is completed.
        """
        return self.task.epoch_done()

    def report(self):
        """
        Report metrics for the subtask.
        """
        self.task.report()

    def reset(self):
        """
        Reset the subtask.
        """
        self.task.reset()

    def reset_metrics(self):
        """
        Reset metrics for the subtask.
        """
        self.task.reset_metrics()

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
        shared['class'] = type(self)
        shared['opt'] = self.opt
        shared['task'] = self.task.share()
        return shared

    def shutdown(self):
        """
        Shutdown the subtask.
        """
        self.task.shutdown()

    def update_counters(self):
        self.task.update_counters()
