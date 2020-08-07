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

from abc import ABC, abstractmethod
from parlai.core.agents import create_agent_from_shared
from parlai.core.opt import Opt
from parlai.core.teachers import create_task_agent_from_taskname, Teacher


class AbstractWrapperTeacher(Teacher, ABC):
    """
    Abstract teacher that will wrap around another teacher and allow for manipulating
    the fields returned by the inner teacher.
    """

    @classmethod
    def add_cmdline_args(cls, parser):
        agent = parser.add_argument_group('AbstractWrapper args')
        agent.add_argument(
            '-wt',
            '--wrapper-task',
            type=str,
            help='The task whose fields will be manipulated.',
        )
        known_args, _ = parser.parse_known_args(nohelp=True)
        parser.add_task_args(known_args.wrapper_task)

    def __init__(self, opt: Opt, shared=None):
        if ',' in opt['task']:
            raise ValueError(
                'AbstractWrapperTeacher cannot be used with multiple tasks!'
            )
        self.id = opt['task']
        self.opt = opt
        if shared:
            self.task = create_agent_from_shared(shared['task'])
        else:
            opt_singletask = copy.deepcopy(opt)
            opt_singletask['task'] = opt['wrapper_task']
            self.task = create_task_agent_from_taskname(opt_singletask)[0]

    @abstractmethod
    def act(self):
        """
        Act on the previous observation.
        """
        raise NotImplementedError('Abstract class: user must implement act() method')

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

    def epoch_done(self):
        """
        Return whether the subtask is completed.
        """
        return self.task.epoch_done()

    def report(self):
        """
        Report metrics for the subtask.
        """
        return self.task.report()

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


class LabelToTextTeacher(AbstractWrapperTeacher):
    """
    Teacher that will shift message['labels'][0] into message['text'] for whatever task
    is specified with --wrapper-task.

    Because the dialogue history is effectively overwritten by this action, all episodes
    will be flattened into one example each.
    """

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)

    def act(self):
        """
        Act on the previous observation.
        """
        act = self.task.act()
        new_act = copy.deepcopy(act)
        if 'labels' in act or 'eval_labels' in act:
            labels_type = 'labels' if 'labels' in act else 'eval_labels'
            labels = act[labels_type]
            if len(labels) != 1:
                raise ValueError('LabelToTextTeacher can only be used with one label!')
            new_act.force_set('text', labels[0])
            new_act.force_set(labels_type, [''])
        else:
            assert 'text' not in act and act['episode_done'] is True
        new_act.force_set('episode_done', True)  # Clear the dialogue history
        return new_act
