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
create_task_agents(str): instantiate task-specific agents (e.g. a teacher)
    from a given task string (e.g. 'babi:task1k:1' or 'squad')
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

def create_task_agent_from_taskname(opt):
    """Creates task agent(s) assuming the input "task_dir:teacher_class"
    e.g. def_string is "babi:Task1k:1"
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
        module_name = "parlai.tasks.%s.agents" % (task)
        my_module = importlib.import_module(module_name)
        teacher_class = getattr(my_module, teacher)
        task_agents = teacher_class(opt)
        if type(task_agents) != list:
            task_agents = [task_agents]
        return task_agents
    else:
        # Multitask teacher/agent
        task_agents = MultiTaskTeacher(opt)
        if type(task_agents) != list:
            task_agents = [task_agents]
        return task_agents


def create_task_agents(opt):
    sp = opt['task'].strip().split(':')
    task = sp[0].lower()
    module_name = "parlai.tasks.%s.agents" % (task)
    my_module = importlib.import_module(module_name)
    try:
        # Tries to call the create_agent function in agents.py
        # of the given task.
        create_agent = getattr(my_module, 'create_agents')
        task_agents = create_agent(opt)
    except:
        # create_agents function does not exist, just look for
        # the agent class defined by the task name directly.
        # (This saves the task creator bothering to define the
        # create_agents function when it is not needed.)
        return create_task_agent_from_taskname(opt)
    if type(task_agents) != list:
        task_agents = [task_agents]
    return task_agents


class MultiTaskTeacher(Teacher):
    """Creates a teacher that is actually a set of teachers each based on
    a task string--each of these teachers will get called in turn,
    either randomly or in order.
    They are all in the same world (they are the same agent switching tasks).

    The task string format is described for the `create_task_agents` function
    above.
    """

    def __init__(self, opt, shared=None):
        self.tasks = []
        tasks = opt['task'].split(',')
        for k in tasks:
            opt_singletask = copy.deepcopy(opt)
            opt_singletask['task'] = k
            self.tasks.extend(create_task_agent_from_taskname(opt_singletask))
        self.task_idx = -1
        self.new_task = True
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
