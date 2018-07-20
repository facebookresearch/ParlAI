# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""This module provides a set of basic agents:

    ``Agent(object)``
    base class for all other agents, implements the ``observe()`` method
    which receives an observation/action dict and the ``act()`` method which
    returns a dict in response.

    ``Teacher(Agent)``
    also implements the ``report()`` method for returning metrics. All ParlAI tasks implement
    the ``Teacher`` class.

    ``MultiTaskTeacher(Teacher)``
    creates a set of teachers based on a task string passed to the ``Teacher``,
    creating multiple teachers within it and alternating between them.

All agents are initialized with the following parameters:

    ``opt`` -- contains any options needed to set up the agent. This generally contains
    all command-line arguments recognized from ``core.params``, as well as other
    options that might be set through the framework to enable certain modes.

    ``shared`` (optional) -- if not ``None``, contains any shared data used to construct
    this particular instantiation of the agent. This data might have been
    initialized by another agent, so that different agents can share the same
    data (possibly in different Processes).

This module also provides a utility method:

    ``create_task_agents(str)``: instantiate task-specific agents (e.g. a teacher)
    from a given task string (e.g. 'babi:task1k:1' or 'squad'). Used by
    ``MultiTaskTeacher``.

"""

from parlai.core.build_data import modelzoo_path
from .metrics import Metrics, aggregate_metrics
import copy
import importlib
import pickle
import random
import os


class Agent(object):
    """Base class for all other agents."""

    def __init__(self, opt, shared=None):
        if not hasattr(self, 'id'):
            self.id = 'agent'
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        self.observation = None

    def observe(self, observation):
        """Receive an observation/action dict."""
        self.observation = observation
        return observation

    def act(self):
        """Return an observation/action dict based upon given observation."""
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

    def epoch_done(self):
        return False

    def reset(self):
        self.observation = None

    def reset_metrics(self):
        pass

    def save(self, path=None):
        """If applicable, save any parameters needed to recreate this agent from
        loaded parameters.
        """
        pass

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


class Teacher(Agent):
    """Basic Teacher agent which keeps track of how many times it's received
    messages. Teachers provide the ``report()`` method to get back metrics."""

    def __init__(self, opt, shared=None):
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        if not hasattr(self, 'id'):
            self.id = opt.get('task', 'teacher')
        if not hasattr(self, 'metrics'):
            if shared and shared.get('metrics'):
                self.metrics = shared['metrics']
            else:
                self.metrics = Metrics(opt)
        self.epochDone = False

    # return state/action dict based upon passed state
    def act(self):
        if self.observation is not None and 'text' in self.observation:
            t = {'text': 'Hello agent!'}
        return t

    def epoch_done(self):
        return self.epochDone

    # Default unknown length
    def num_examples(self):
        return None

    def num_episodes(self):
        return None

    # Return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        return self.metrics.report()

    def reset(self):
        super().reset()
        self.reset_metrics()
        self.epochDone = False

    def reset_metrics(self):
        self.metrics.clear()

    def share(self):
        """In addition to default Agent shared parameters, share metrics."""
        shared = super().share()
        shared['metrics'] = self.metrics
        return shared


class MultiTaskTeacher(Teacher):
    """Creates a teacher that is actually a set of teachers each based on
    a task string--each of these teachers will get called in turn,
    either randomly or in order.
    They are all in the same world (they are the same agent switching tasks).

    The task string format is described for the ``create_task_agents()`` function
    above.
    """

    def __init__(self, opt, shared=None):
        self.tasks = []
        self.opt = opt

        opt['batch_sort'] = False
        print('WARNING: batch_sort disabled for multitasking')

        self.id = opt['task']
        if shared and 'tasks' in shared:
            self.tasks = [create_agent_from_shared(t) for t in shared['tasks']]
        else:
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

    def num_examples(self):
        if not hasattr(self, 'num_exs'):
            # num_examples is sum of all examples in all tasks
            tasks_num_exs = [t.num_examples() for t in self.tasks]
            if any(num is None for num in tasks_num_exs):
                self.num_exs = None
            else:
                self.num_exs = sum(tasks_num_exs)
        return self.num_exs

    def num_episodes(self):
        if not hasattr(self, 'num_eps'):
            # num_episodes is sum of all num_episodes in all tasks
            tasks_num_eps = [t.num_episodes() for t in self.tasks]
            if any(num is None for num in tasks_num_eps):
                self.num_eps = None
            else:
                self.num_eps = sum(tasks_num_eps)
        return self.num_eps

    def observe(self, observation):
        return self.tasks[self.task_idx].observe(observation)

    def act(self):
        if self.new_task:
            self.new_task = False
            if self.random:
                # select random teacher
                self.task_idx = random.randrange(len(self.tasks))
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
        for t in self.tasks:
            if not t.epoch_done():
                return False
        return True

    # return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        return aggregate_metrics(self.tasks)

    def reset(self):
        for t in self.tasks:
            t.reset()

    def reset_metrics(self):
        for t in self.tasks:
            t.reset_metrics()

    def save(self):
        for t in self.tasks:
            t.save()

    def share(self):
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        shared['tasks'] = [t.share() for t in self.tasks]
        return shared

    def shutdown(self):
        """Shutdown each agent."""
        for t in self.tasks:
            t.shutdown()


def name_to_agent_class(name):
    """Convert agent name to class.

    This adds "Agent" to the end of the name and uppercases the first letter
    and the first letter appearing after each underscore (underscores are
    removed).

    :param name: name of agent, e.g. local_human

    Returns class of agent, e.g. LocalHumanAgent.
    """
    words = name.split('_')
    class_name = ''
    for w in words:
        class_name += (w[0].upper() + w[1:])
    class_name += 'Agent'
    return class_name


def load_agent_module(opt):
    """Load agent options and module from file if opt file exists.

    Checks to see if file exists opt['model_file'] + ".opt"; if so, load up the
    options from the file and use that to create an agent, loading the model
    type from that file and overriding any options specified in that file when
    instantiating the agent.

    If that file does not exist, return None.
    """
    model_file = opt['model_file']
    optfile = model_file + '.opt'
    if os.path.isfile(optfile):
        with open(optfile, 'rb') as handle:
            new_opt = pickle.load(handle)
        if 'batchindex' in new_opt:
            # This saved variable can cause trouble if we switch to BS=1 at test time
            del new_opt['batchindex']
        # only override opts specified in 'override' dict
        if opt.get('override'):
            for k, v in opt['override'].items():
                if str(v) != str(new_opt.get(k, None)):
                    print("[ warning: overriding opt['{}'] to {} ("
                          "previously: {} )]".format(
                            k, v, new_opt.get(k, None)))
                new_opt[k] = v
        # add model arguments to new_opt if they aren't in new_opt already
        for k, v in opt.items():
            if k not in new_opt:
                new_opt[k] = v
        new_opt['model_file'] = model_file
        model_class = get_agent_module(new_opt['model'])
        return model_class(new_opt)
    else:
        return None


def get_agent_module(dir_name):
    """Return the module for an agent specified by `--model`.

    Can be formatted in several different ways:
    - full: -m parlai.agents.seq2seq.seq2seq:Seq2seqAgent
    - shorthand: -m seq2seq, which will check both paths
        parlai.agents.seq2seq.seq2seq:Seq2seqAgent and
        parlai.agents.seq2seq.agents:Seq2seqAgent
    - half-shorthand: -m seq2seq/variant, which will check the path
        parlai.agents.seq2seq.variant:VariantAgent

    The base path to search when using shorthand formats can be changed from
    "parlai" to "parlai_internal" by prepending "internal:" to the path, e.g.
    "internal:seq2seq".

    :param dir_name: path to model class in one of the above formats.
    """
    repo = 'parlai'
    if dir_name.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        # this will follow the same paths but look in parlai_internal instead
        repo = 'parlai_internal'
        dir_name = dir_name[9:]
    if ':' in dir_name:
        # e.g. -m "parlai.agents.seq2seq.seq2seq:Seq2seqAgent"
        s = dir_name.split(':')
        module_name = s[0]
        class_name = s[1]
    elif '/' in dir_name:
        # e.g. -m my_agent/special_variant
        # will check parlai.agents.my_agent.special_variant:SpecialVariantAgent
        sp = dir_name.split('/')
        module_name = "%s.agents.%s.%s" % (repo, sp[0], sp[1])
        class_name = name_to_agent_class(sp[1])
    else:
        # e.g. -m seq2seq
        # will check parlai.agents.seq2seq.agents for Seq2seqAgent first
        # then check parlai.agents.seq2seq.seq2seq for Seq2seqAgent second
        class_name = name_to_agent_class(dir_name)
        try:
            module_name = "%s.agents.%s.agents" % (repo, dir_name)
            importlib.import_module(module_name)  # check if it's there
        except ImportError:
            module_name = "%s.agents.%s.%s" % (repo, dir_name, dir_name)
    my_module = importlib.import_module(module_name)
    model_class = getattr(my_module, class_name)
    return model_class


def create_agent(opt, requireModelExists=False):
    """Create an agent from the options ``model``, ``model_params`` and ``model_file``.
    The input is either of the form ``parlai.agents.ir_baseline.agents:IrBaselineAgent``
    (i.e. the path followed by the class name) or else just ``ir_baseline`` which
    assumes the path above, and a class name suffixed with 'Agent'.

    If ``model-file`` is available in the options this function can also attempt to load
    the model from that location instead. This avoids having to specify all the other
    options necessary to set up the model including its name as they are all loaded from
    the options file if it exists (the file opt['model_file'] + '.opt' must exist and
    contain a pickled dict containing the model's options).
    """
    if opt.get('datapath', None) is None:
        # add datapath, it is missing
        from parlai.core.params import ParlaiParser, get_model_name
        parser = ParlaiParser(add_parlai_args=False)
        parser.add_parlai_data_path()
        # add model args if they are missing
        model = get_model_name(opt)
        if model is not None:
            parser.add_model_subargs(model)
        opt_parser = parser.parse_args("", print_args=False)
        for k, v in opt_parser.items():
            if k not in opt:
                opt[k] = v

    if opt.get('model_file'):
        opt['model_file'] = modelzoo_path(opt.get('datapath'), opt['model_file'])
        if requireModelExists and not os.path.isfile(opt['model_file']):
            raise RuntimeError('WARNING: Model file does not exist, check to make '
                               'sure it is correct: {}'.format(opt['model_file']))
        # Attempt to load the model from the model file first (this way we do
        # not even have to specify the model name as a parameter)
        model = load_agent_module(opt)
        if model is not None:
            return model
        else:
            print("[ no model with opt yet at: " + opt.get('model_file') + "(.opt) ]")

    if opt.get('model'):
        model_class = get_agent_module(opt['model'])
        model = model_class(opt)
        if requireModelExists and hasattr(model, 'load') and not opt.get('model_file'):
            # double check that we didn't forget to set model_file on loadable model
            print('WARNING: model_file unset but model has a `load` function.')
        return model
    else:
        raise RuntimeError('Need to set `model` argument to use create_agent.')


# Helper functions to create agent/agents given shared parameters
# returned from agent.share(). Useful for parallelism, sharing params, etc.
def create_agent_from_shared(shared_agent):
    """Instantiate an agent from the default `shared` params.

    :param shared_agent: should include an `opt` dictionary and agent `class`,
        along with whatever other parameters the agent needs to instantiate.
    """
    opt = copy.deepcopy(shared_agent['opt'])  # TODO: is this slow sometimes?
    a = shared_agent['class'](opt, shared_agent)
    return a


def create_agents_from_shared(shared):
    """Create agents based on shared data.

    :param shared: `list` of `dict` objects created by calling e.g.
        [a.share() for a in agents].

    Returns a list of instantiated agents.
    """
    shared_agents = []
    for shared_agent in shared:
        agent = create_agent_from_shared(shared_agent)
        shared_agents.append(agent)
    return shared_agents


def get_task_module(taskname):
    """Get the module of the task agent specified by `--task`.

    Can be formatted in several different ways:
    - full: -t parlai.tasks.babi.agents:DefaultTeacher
    - shorthand: -t babi, which will check
        parlai.tasks.babi.agents:DefaultTeacher
    - shorthand specific: -t babi:task10k, which will check
        parlai.tasks.babi.agents:Task10kTeacher

    The base path to search when using shorthand formats can be changed from
    "parlai" to "parlai_internal" by prepending "internal:" to the path, e.g.
    "internal:babi".

    Options can be sent to the teacher by adding an additional colon,
    for example "-t babi:task10k:1" directs the babi Task10kTeacher to use
    task number 1.

    :param taskname: path to task class in one of the above formats.
    """
    sp = taskname.strip()
    repo = 'parlai'
    if sp.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        repo = 'parlai_internal'
        sp = sp[9:]
    sp = sp.split(':')
    if '.' in sp[0]:
        module_name = sp[0]
    elif sp[0] == 'pytorch_teacher':
        module_name = 'parlai.core.pytorch_data_teacher'
    else:
        task = sp[0].lower()
        module_name = "%s.tasks.%s.agents" % (repo, task)
    if len(sp) > 1:
        sp[1] = sp[1][0].upper() + sp[1][1:]
        teacher = sp[1]
        if '.' not in sp[0] and 'Teacher' not in teacher:
            # Reformat from underscore to CamelCase and append "Teacher" to
            # class name by default if a complete path is not given.
            words = teacher.split('_')
            teacher_name = ''
            for w in words:
                teacher_name += (w[0].upper() + w[1:])
            teacher = teacher_name + "Teacher"
    else:
        teacher = "DefaultTeacher"
    my_module = importlib.import_module(module_name)
    teacher_class = getattr(my_module, teacher)
    return teacher_class


def create_task_agent_from_taskname(opt):
    """Create task agent(s) assuming the input ``task_dir:teacher_class``.

    e.g. def_string is a shorthand path like ``babi:Task1k:1`` or ``#babi``
    or a complete path like ``parlai.tasks.babi.agents:Task1kTeacher:1``,
    which essentially performs ``from parlai.tasks.babi import Task1kTeacher``
    with the parameter ``1`` in ``opt['task']`` to be used by the class ``Task1kTeacher``.
    """
    if not (opt.get('task') or opt.get('pytorch_teacher_task') or opt.get('pytorch_teacher_dataset')):
        raise RuntimeError('No task specified. Please select a task with ' +
                           '--task {task_name}.')
    if not opt.get('task'):
        opt['task'] = 'pytorch_teacher'
    if ',' not in opt['task']:
        # Single task
        teacher_class = get_task_module(opt['task'])
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
    """Create task agent(s) for the given task name.

    It does this by calling the create_agent function in agents.py of the
    given task.
    If create_agents function does not exist, it just looks for
    the teacher (agent) class defined by the task name directly.
    (This saves the task creator bothering to define the
    create_agents function when it is not needed.)
    """
    sp = opt['task'].strip()
    repo = 'parlai'
    if sp.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        repo = 'parlai_internal'
        sp = sp[9:]
    sp = sp.split(':')
    if '.' in sp[0]:
        # The case of opt['task'] = 'parlai.tasks.squad.agents:DefaultTeacher'
        # (i.e. specifying your own path directly)
        module_name = sp[0]
    elif sp[0] == 'pytorch_teacher':
        module_name = 'parlai.core.pytorch_data_teacher'
    else:
        task = sp[0].lower()
        module_name = "%s.tasks.%s.agents" % (repo, task)
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
