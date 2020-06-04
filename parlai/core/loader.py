#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Functions for loading modules for agents, tasks and teachers, and worlds.

These functions are largely for converting strings specified in opts (like for --task)
to the appropriate module.
"""

from typing import Callable, Dict, Type
import importlib


##############################################################
### REGISTRY
##############################################################
# for user added agents needed in just one script, or similar

AGENT_REGISTRY: Dict[str, Type] = {}
TEACHER_REGISTRY: Dict[str, Type] = {}


def register_agent(name: str) -> Callable[[Type], Type]:
    """
    Register an agent to be available in command line calls.

    >>> @register_teacher("my_agent")
    ... class MyAgent:
    ...     pass
    """

    def _inner(cls_):
        global AGENT_REGISTRY
        AGENT_REGISTRY[name] = cls_
        return cls_

    return _inner


def register_teacher(name: str) -> Callable[[Type], Type]:
    """
    Register a teacher to be available as a command line.

    >>> @register_teacher("my_teacher")
    ... class MyTeacher:
    ...    pass
    """

    def _inner(cls_):
        global TEACHER_REGISTRY
        TEACHER_REGISTRY[name] = cls_
        return cls_

    return _inner


##############################################################
### AGENT LOADER
##############################################################
def _name_to_agent_class(name: str):
    """
    Convert agent name to class.

    This adds "Agent" to the end of the name and uppercases the first letter
    and the first letter appearing after each underscore (underscores are
    removed).

    :param name:
        name of agent, e.g. local_human

    :return:
        class of agent, e.g. LocalHumanAgent.
    """
    words = name.split('_')
    class_name = ''
    for w in words:
        # capitalize the first letter
        class_name += w[0].upper() + w[1:]
    # add Agent to the end of the name
    class_name += 'Agent'
    return class_name


def load_agent_module(agent_path: str):
    """
    Return the module for an agent specified by ``--model``.

    Can be formatted in several different ways:

    * full: `-m parlai.agents.seq2seq.seq2seq:Seq2seqAgent`
    * shorthand: -m seq2seq, which will check both paths
      ``parlai.agents.seq2seq.seq2seq:Seq2seqAgent`` and
      ``parlai.agents.seq2seq.agents:Seq2seqAgent``
    * half-shorthand: ``-m seq2seq/variant``, which will check the path
      `parlai.agents.seq2seq.variant:VariantAgent`
    * legacy models: ``-m legacy:seq2seq:0``, which will look for the deprecated
      model at ``parlai.agents.legacy_agents.seq2seq.seq2seq_v0:Seq2seqAgent``

    The base path to search when using shorthand formats can be changed from
    "parlai" to "parlai_internal" by prepending "internal:" to the path, e.g.
    "internal:seq2seq".

    To use legacy agent versions, you can prepend "legacy:" to model arguments,
    e.g. "legacy:seq2seq:0" will translate to ``legacy_agents/seq2seq/seq2seq_v0``.

    To use agents in projects, you can prepend "projects:" and the name of the
    project folder to model arguments, e.g. "projects:personachat:kvmemnn"
    will translate to ``projects/personachat/kvmemnn``.

    :param agent_path:
        path to model class in one of the above formats.

    :return:
        module of agent
    """
    global AGENT_REGISTRY
    if agent_path in AGENT_REGISTRY:
        return AGENT_REGISTRY[agent_path]

    repo = 'parlai'
    if agent_path.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        # this will follow the same paths but look in parlai_internal instead
        repo = 'parlai_internal'
        agent_path = agent_path[9:]
    elif agent_path.startswith('fb:'):
        repo = 'parlai_fb'
        agent_path = agent_path[3:]

    if agent_path.startswith('legacy:'):
        # e.g. -m legacy:seq2seq:0
        # will check legacy_agents.seq2seq.seq2seq_v0:Seq2seqAgent
        path_list = agent_path.split(':')
        if len(path_list) != 3:
            raise RuntimeError(
                'legacy paths should follow pattern '
                'legacy:model:version; you used {}'
                ''.format(agent_path)
            )
        model_name = path_list[1]  # seq2seq
        module_name = 'parlai.agents.legacy_agents.{m}.{m}_v{v}'.format(
            m=model_name, v=path_list[2]
        )
        class_name = _name_to_agent_class(model_name)
    elif agent_path.startswith('projects:'):
        # e.g. -m projects:personachat:kvmemnn
        path_list = agent_path.split(':')
        if len(path_list) != 3:
            raise RuntimeError(
                'projects paths should follow pattern '
                'projects:folder:model; you used {}'
                ''.format(agent_path)
            )
        folder_name = path_list[1]
        model_name = path_list[2]
        module_name = 'projects.{p}.{m}.{m}'.format(m=model_name, p=folder_name)
        class_name = _name_to_agent_class(model_name)
    elif ':' in agent_path:
        # e.g. -m "parlai.agents.seq2seq.seq2seq:Seq2seqAgent"
        path_list = agent_path.split(':')
        module_name = path_list[0]
        class_name = path_list[1]
    elif '/' in agent_path:
        # e.g. -m my_agent/special_variant
        # will check parlai.agents.my_agent.special_variant:SpecialVariantAgent
        path_list = agent_path.split('/')
        module_name = "%s.agents.%s.%s" % (repo, path_list[0], path_list[1])
        class_name = _name_to_agent_class(path_list[1])
    else:
        # e.g. -m seq2seq
        # will check parlai.agents.seq2seq.agents for Seq2seqAgent first
        # then check parlai.agents.seq2seq.seq2seq for Seq2seqAgent second
        class_name = _name_to_agent_class(agent_path)
        try:
            module_name = "%s.agents.%s.agents" % (repo, agent_path)
            importlib.import_module(module_name)  # check if it's there
        except ImportError:
            module_name = "%s.agents.%s.%s" % (repo, agent_path, agent_path)

    my_module = importlib.import_module(module_name)
    model_class = getattr(my_module, class_name)

    return model_class


##############################################################
### TASK AND TEACHER LOADERS
##############################################################
def _get_task_path_and_repo(taskname: str):
    """
    Returns the task path list and repository containing the task as specified by
    `--task`.

    :param taskname: path to task class (specified in format detailed below)
    """
    task = taskname.strip()
    repo = 'parlai'
    if task.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        repo = 'parlai_internal'
        task = task[9:]
    elif task.startswith('fb:'):
        repo = 'parlai_fb'
        task = task[3:]

    task_path_list = task.split(':')

    return task_path_list, repo


def load_task_module(taskname: str):
    """
    Get the module containing all teacher agents for the task specified by `--task`.

    :param taskname: path to task class in one of the following formats:
        * full: ``-t parlai.tasks.babi.agents:DefaultTeacher``
        * shorthand: ``-t babi``, which will check
            ``parlai.tasks.babi.agents:DefaultTeacher``
        * shorthand specific: ``-t babi:task10k``, which will check
            ``parlai.tasks.babi.agents:Task10kTeacher``

    :return:
        module containing all teacher agents for a task
    """
    task_path_list, repo = _get_task_path_and_repo(taskname)
    task_path = task_path_list[0]

    if '.' in task_path:
        module_name = task_path
    else:
        task = task_path.lower()
        module_name = "%s.tasks.%s.agents" % (repo, task)

    task_module = importlib.import_module(module_name)

    return task_module


def load_teacher_module(taskname: str):
    """
    Get the module of the teacher agent specified by `--task`.

    Can be formatted in several different ways:

    * full: ``-t parlai.tasks.babi.agents:DefaultTeacher``
    * shorthand: ``-t babi``, which will check
        ``parlai.tasks.babi.agents:DefaultTeacher``
    * shorthand specific: ``-t babi:task10k``, which will check
        ``parlai.tasks.babi.agents:Task10kTeacher``

    The base path to search when using shorthand formats can be changed from
    "parlai" to "parlai_internal" by prepending "internal:" to the path, e.g.
    "internal:babi".

    Options can be sent to the teacher by adding an additional colon,
    for example ``-t babi:task10k:1`` directs the babi Task10kTeacher to use
    task number 1.

    :param taskname: path to task class in one of the above formats.

    :return:
        teacher module
    """
    global TEACHER_REGISTRY
    if taskname in TEACHER_REGISTRY:
        return TEACHER_REGISTRY[taskname]

    task_module = load_task_module(taskname)
    task_path_list, repo = _get_task_path_and_repo(taskname)

    if len(task_path_list) > 1 and '=' not in task_path_list[1]:
        task_path_list[1] = task_path_list[1][0].upper() + task_path_list[1][1:]
        teacher = task_path_list[1]
        if '.' not in task_path_list[0] and 'Teacher' not in teacher:
            # Reformat from underscore to CamelCase and append "Teacher" to
            # class name by default if a complete path is not given.
            words = teacher.split('_')
            teacher_name = ''
            for w in words:
                teacher_name += w[0].upper() + w[1:]
            teacher = teacher_name + "Teacher"
    else:
        teacher = "DefaultTeacher"

    teacher_class = getattr(task_module, teacher)
    return teacher_class


##############################################################
### WORLD LOADER
##############################################################
def _get_default_world(default_world=None, num_agents=None):
    """
    Get default world if a world is not already specified by the task.

    If a default world is provided, return this. Otherwise, return
    DialogPartnerWorld if there are 2 agents and MultiAgentDialogWorld if
    there are more.

    :param default_world:
        default world to return
    :param num_agents:
        number of agents in the environment
    """
    if default_world is not None:
        world_class = default_world
    elif num_agents is not None:
        import parlai.core.worlds as core_worlds

        world_name = (
            "DialogPartnerWorld" if num_agents == 2 else "MultiAgentDialogWorld"
        )
        world_class = getattr(core_worlds, world_name)
    else:
        return None

    return world_class


def load_world_module(
    taskname: str,
    interactive_task: bool = False,
    selfchat_task: bool = False,
    num_agents: int = None,  # a priori may not know the number of agents
    default_world=None,
):
    """
    Load the world module for the specific environment. If not enough information is to
    determine which world should be loaded, returns None.

    :param taskname:
        path to task class in one of the above formats
    :param interactive_task:
        whether or not the task is interactive
    :param num_agents:
        number of agents in the world; this may not be known a priori
    :param default_world:
        default world to return if specified

    :return:
        World module (or None, if not enough info to determine is present)
    """
    task = taskname.strip()
    repo = 'parlai'
    if task.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        repo = 'parlai_internal'
        task = task[9:]
    task_path_list = task.split(':')
    if '.' in task_path_list[0]:
        # The case of opt['task'] = 'parlai.tasks.squad.agents:DefaultTeacher'
        # (i.e. specifying your own path directly, assumes DialogPartnerWorld)
        return _get_default_world(default_world, num_agents)

    task = task_path_list[0].lower()
    if len(task_path_list) > 1:
        task_path_list[1] = task_path_list[1][0].upper() + task_path_list[1][1:]
        world_name = task_path_list[1] + "World"
        if interactive_task:
            world_name = "Interactive" + world_name
        elif selfchat_task:
            world_name = "SelfChat" + world_name
    else:
        if interactive_task:
            world_name = "InteractiveWorld"
        elif selfchat_task:
            world_name = "SelfChatWorld"
        else:
            world_name = "DefaultWorld"
    module_name = "%s.tasks.%s.worlds" % (repo, task)

    try:
        my_module = importlib.import_module(module_name)
        world_class = getattr(my_module, world_name)
    except (ModuleNotFoundError, AttributeError):
        # Defaults to this if you did not specify a world for your task.
        world_class = _get_default_world(default_world, num_agents)

    return world_class
