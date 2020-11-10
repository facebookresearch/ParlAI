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

from abc import ABC
from parlai.core.agents import create_agent_from_shared
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.teachers import (
    create_task_agent_from_taskname,
    FixedDialogTeacher,
    Teacher,
)
from parlai.utils.misc import warn_once


class AbstractWrapperTeacher(Teacher, ABC):
    """
    Abstract teacher that wraps around another teacher.

    This teacher allows for manipulating the fields returned by the inner teacher, in
    the abstract self._edit_action() method that is called during self.act(). The inner
    teacher must subclass FixedDialogTeacher in order to make use of that teacher's
    .get_orig_action() and .process_action() methods.
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
        try:
            parser.add_task_args(known_args.wrapper_task)
        except RuntimeError:
            warn_once(
                'The task name cannot be parsed from command-line arguments! '
                'Task-specific flags will not be added.'
            )

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
        assert isinstance(self.task, FixedDialogTeacher)

    def act(self):
        """
        Act on the previous observation.

        Normally, the inner teacher would call .get_orig_action() and .process_action();
        here, we insert an ._edit_action() method in between these two methods in order
        to allow for arbitrary manipulation of the action before it is registered and
        processed further by the inner teacher.
        """
        orig_action = self.task.get_orig_action()
        edited_action = self._edit_action(orig_action)
        processed_action = self.task.process_action(edited_action)
        return processed_action

    def _edit_action(self, act: Message) -> Message:
        """
        Edit and return the input action.

        The input action typically comes from the inner teacher's .get_orig_action()
        method.
        """
        raise NotImplementedError(
            'Abstract class: user must implement the _edit_action() method'
        )

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

    def _edit_action(self, act: Message) -> Message:
        """
        Edit the fields of the action manually.
        """
        if 'labels' in act:
            labels = act['labels']
            if len(labels) != 1:
                raise ValueError(
                    f'{type(self).__name__} can only be used with one label!'
                )
            act.force_set('text', labels[0])
            act.force_set('labels', [''])
        else:
            assert 'text' not in act and act['episode_done'] is True
        act.force_set('episode_done', True)  # Clear the dialogue history
        return act
