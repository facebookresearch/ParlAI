# Copyright 2004-present Facebook. All Rights Reserved.
"""Provides a set of basic agents:

Agent(object): base class for all other agents, implements the act() method
    which receives an observation table and returns a table in response
Teacher(Agent): also implements the report method for returning metrics. Tasks
    implement the Teacher class.
MultiTaskTeacher(Teacher): creates a set of teachers based on a "task string"
    passed to the Teacher, creating multiple teachers within it and alternating
    between them

Also provides a utility method (used by MultiTaskTeacher) for instantiating
    teachers from a string, assuming they follow our naming conventions:
create_task_teacher(str): instantiate a teacher from a given task string
    (e.g. 'babi:task:1' or 'squad')
"""

import copy
import importlib
import random


class Agent(object):
    """Basic agent which says hello."""

    def __init__(self, opt, shared=None):
        print('agent initializing!')

    def act(self, observation):
        """Return state/action table based upon given observation."""
        if observation is not None:
            print('agent received observation:')
            print(observation)

        t = {}
        t['text'] = 'hello, teacher!'
        print('agent sending message:')
        print(t)
        return t

    def share(self, opt):
        """If applicable, share any parameters needed to create a shared version
        of this agent.
        """
        pass

    def shutdown(self):
        """Perform any final cleanup if needed."""
        pass


class Teacher(Agent):
    """Basic Teacher agent which keeps track of how many times it's received
    messages. Teachers provide the `report` method to get back metrics."""

    def __init__(self, opt, shared=None):
        print('teacher initializing!')
        self.metrics = {
            'text_received': 0,
        }

    # return state/action dict based upon passed state
    def act(self, observation):
        if observation is not None and 'text' in observation:
            self.metrics['text_received'] += 1

        t = {
            'text': 'Hello agent. I have heard from you {0} times'.format(
                self.metrics['text_received']
            )
        }
        return t

    def report(self):
        return self.metrics


def create_task_teacher(opt):
    """Creates task agent(s) assuming the input "task_dir:teacher_class"
    e.g. def_string is "babi:Task:1"
    This essentially performs "from parlai.tasks.babi import TaskTeacher"
    with the parameter 1 in opt['task'] to be used by the class TaskTeacher
    """
    if ',' not in opt['task']:
        # Single task
        sp = opt['task'].strip().split(':')
        task = sp[0].lower()
        if len(sp) > 1:
            sp[1] = sp[1][0].upper() + sp[1][1:]
            teacher = sp[1] + "Teacher"
        else:
            teacher = "DefaultTeacher"
        module_name = "parlai.tasks.%s.teachers" % (task)
        my_module = importlib.import_module(module_name)
        class_name = teacher
        teacher_class = getattr(my_module, class_name)
        return teacher_class(opt)
    else:
        # Multitask
        teacher_class = MultiTaskTeacher
        return teacher_class(opt)


class MultiTaskTeacher(Teacher):
    """Generates a set of teachers based on a task string--each of these
    teachers will get called in turn, either randomly or in order.

    The task string format is described for the `create_task_teacher` function
    above.
    """

    def __init__(self, opt, shared=None):
        self.tasks = []
        tasks = opt['task'].split(',')
        for k in tasks:
            print("[creating " + k + "]")
            opt_singletask = copy.deepcopy(opt)
            opt_singletask['task'] = k
            self.tasks.append(create_task_teacher(opt_singletask))
        self.task_idx = -1
        self.new_task = False
        self.random = opt.get('datatype') == 'train'

    def __len__(self):
        if not hasattr(self, 'len'):
            self.len = 0
            # length is sum of all task lengths
            for _ind, t in enumerate(self.tasks):
                self.len += len(t)
        return self.len

    def act(self, observation):
        if self.new_task:
            self.new_task = False
            if self.random:
                self.task_idx = random.randrange(len(self.tasks))
            else:
                self.task_idx = (self.task_idx + 1) % len(self.tasks)
        t = self.tasks[self.task_idx].act(observation)
        if t['done']:
            self.new_task = True
        return t

    # return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        m = {}
        # TODO(jase): fix metrics, add them up or?
        return m
