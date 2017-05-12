# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Provides a set of basic agents:

Agent(object): base class for all other agents, implements the observe() method
    method  which receives an observation table and the act method which
    returns a table in response
Teacher(Agent): also implements the report method for returning metrics. Tasks
    implement the Teacher class.
MultiTaskTeacher(Teacher): creates a set of teachers based on a "task string"
    passed to the Teacher, creating multiple teachers within it and alternating
    between them

Also provides a utility method (used by MultiTaskTeacher) for instantiating
    teachers from a string, assuming they follow our naming conventions:
create_task_agents(str): instantiate task-specific agents (e.g. a teacher)
    from a given task string (e.g. 'babi:task1k:1' or 'squad')


All agents are initialized with the following parameters:

opt -- contains any options needed to set up the agent. This generally contains
    all command-line arguments recognized from core.params, as well as other
    options that might be set through the framework to enable certain modes.
shared (optional) -- if not None, contains any shared data used to construct
    this particular instantiation of the agent. This data might have been
    initialized by another agent, so that different agents can share the same
    data (possibly in different Processes).
"""

from .metrics import Metrics
import copy
import importlib
import random


class Agent(object):
    """Basic agent which says hello."""

    def __init__(self, opt, shared=None):
        if not hasattr(self, 'id'):
            self.id = 'agent'
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        self.observation = None

    def observe(self, observation):
        self.observation = observation
        return observation

    def act(self):
        """Return state/action table based upon given observation."""
        if hasattr(self, 'observation') and self.observation is not None:
            print('agent received observation:')
            print(self.observation)

        t = {}
        t['text'] = 'hello, teacher!'
        print('agent sending message:')
        print(t)
        return t

    def getID(self):
        return self.id

    def reset(self):
        self.observation = None

    def share(self):
        """If applicable, share any parameters needed to create a shared version
        of this agent.
        """
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        return shared

    def shutdown(self):
        """Perform any final cleanup if needed."""
        pass

def create_agent(opt):
    """Create an agent from the options model, model_params and model_file.
    The input is either of the form "parlai.agents.ir_baseline.agents/IrBaselineAgent"
    (i.e. the path followed by the class name) or else just 'IrBaseline' which
    assumes the path above, and a class name suffixed with 'Agent'
    """
    dir_name = opt['model']
    if ':' in dir_name:
        s = dir_name.split(':')
        module_name = s[0]
        class_name = s[1]
    else:
        module_name = "parlai.agents.%s.agents" % (dir_name)
        words = opt['model'].split('_')
        class_name = ''
        for w in words:
            class_name += ( w[0].upper() + w[1:])
        class_name += 'Agent'
    print(class_name)
    my_module = importlib.import_module(module_name)
    model_class = getattr(my_module, class_name)
    return model_class(opt)

# Helper functions to create agent/agents given shared parameters
# returned from agent.share(). Useful for parallelism, sharing params, etc.
def create_agent_from_shared(shared_agent):
    a = shared_agent['class'](shared_agent['opt'], shared_agent)
    return a

def create_agents_from_shared(shared):
    # create agents based on shared data.
    shared_agents = []
    for shared_agent in shared:
        agent = create_agent_from_shared(shared_agent)
        shared_agents.append(agent)
    return shared_agents


class Teacher(Agent):
    """Basic Teacher agent which keeps track of how many times it's received
    messages. Teachers provide the `report` method to get back metrics."""

    def __init__(self, opt, shared=None):
        if not hasattr(self, 'opt'):
            self.opt = opt
        if not hasattr(self, 'id'):
            self.id = opt.get('task', 'teacher')
        if not hasattr(self, 'metrics'):
            if shared and shared.get('metrics'):
                self.metrics = shared['metrics']
            else:
                self.metrics = Metrics(opt)
        self.epochDone = False

    def __iter__(self):
        """Teacher can be iterated over. Subclasses can specify a certain length
        of iteration, such as e.g. one epoch.
        """
        self.epochDone = False
        return self

    def __next__(self):
        """Raise StopIteration if epoch is done (never for default teacher)."""
        if self.epochDone:
            raise StopIteration()

    # return state/action dict based upon passed state
    def act(self):
        if self.observation is not None and 'text' in self.observation:
            t = { 'text': 'Hello agent!' }
        return t

    def epoch_done(self):
        return self.epochDone

    # Return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        report = self.metrics.report()
        return report

    def reset(self):
        super().reset()
        self.epochDone = False
        self.metrics.clear()

    def share(self):
        """If applicable, share any parameters needed to create a shared version
        of this agent.
        """
        shared = super().share()
        shared['metrics'] = self.metrics
        return shared


def create_task_agent_from_taskname(opt):
    """Creates task agent(s) assuming the input "task_dir:teacher_class"
    e.g. def_string is a shorthand path like "babi:Task1k:1" or "#babi"
    or a complete path like "parlai.tasks.babi.agents:Task1kTeacher:1"
    This essentially performs "from parlai.tasks.babi import Task1kTeacher"
    with the parameter 1 in opt['task'] to be used by the class Task1kTeacher
    """
    if ',' not in opt['task']:
        # Single task
        sp = opt['task'].strip().split(':')
        if '.' in sp[0]:
            module_name = sp[0]
        else:
            task = sp[0].lower()
            module_name = "parlai.tasks.%s.agents" % (task)
        if len(sp) > 1:
            sp[1] = sp[1][0].upper() + sp[1][1:]
            teacher = sp[1]
            if '.' not in sp[0] and 'Teacher' not in teacher:
                # Append "Teacher" to class name by default if
                # a complete path is not given.
                teacher += "Teacher"
        else:
            teacher = "DefaultTeacher"
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


def _create_task_agents(opt):
    """Creates task agent(s) for the given task name.
    It does this by calling the create_agent function in agents.py of the
    given task.
    If create_agents function does not exist, it just looks for
    the teacher (agent) class defined by the task name directly.
    (This saves the task creator bothering to define the
    create_agents function when it is not needed.)
    """
    sp = opt['task'].strip().split(':')
    if '.' in sp[0]:
        # The case of opt['task'] = 'parlai.tasks.squad.agents:DefaultTeacher'
        # (i.e. specifying your own path directly)
        module_name = sp[0]
    else:
        task = sp[0].lower()
        module_name = "parlai.tasks.%s.agents" % (task)
    my_module = importlib.import_module(module_name)
    try:
        # Tries to call the create_agent function in agents.py
        create_agent = getattr(my_module, 'create_agents')
        task_agents = create_agent(opt)
    except AttributeError:
        # Create_agent not found, so try to create the teacher directly.
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
        self.opt = opt
        self.id = opt['task']
        tasks = opt['task'].split(',')
        for k in tasks:
            k = k.strip()
            if k:
                opt_singletask = copy.deepcopy(opt)
                opt_singletask['task'] = k
                self.tasks.extend(create_task_agent_from_taskname(
                    opt_singletask))
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

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch_done():
            raise StopIteration()

    def observe(self, observation):
        self.tasks[self.task_idx].observe(observation)
        if self.new_task:
            self.new_task = False
            if self.random:
                self.task_idx = random.randrange(len(self.tasks))
            else:
                start_idx = self.task_idx
                keep_looking = True
                while keep_looking:
                    self.task_idx = (self.task_idx + 1) % len(self.tasks)
                    keep_looking = (self.tasks[self.task_idx].epoch_done() and
                                    start_idx != self.task_idx)
                if start_idx == self.task_idx:
                    return {'text': 'There are no more examples remaining.'}
        return observation

    def act(self):
        t = self.tasks[self.task_idx].act()
        if t['episode_done']:
            self.new_task = True
        return t

    def epoch_done(self):
        for t in self.tasks:
            if not t.epoch_done():
                return False
        return True

    # return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        m = {}
        m['tasks'] = {}
        sum_accuracy = 0
        num_tasks = 0
        total = 0
        for i in range(len(self.tasks)):
            mt = self.tasks[i].report()
            m['tasks'][self.tasks[i].getID()] = mt
            total += mt['total']
            if 'accuracy' in mt:
                sum_accuracy += mt['accuracy']
                num_tasks += 1
        m['total'] = total
        m['accuracy'] = 0
        if num_tasks > 0:
            m['accuracy'] = sum_accuracy / num_tasks
        return m
