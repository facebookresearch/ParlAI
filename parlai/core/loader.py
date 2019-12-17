#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Functions for loading modules for tasks, agents, and worlds.
TODO: make this description nicer
"""

import importlib


# AGENT LOADER
def name_to_agent_class(name: str):
    """
    Convert agent name to class.

    This adds "Agent" to the end of the name and uppercases the first letter
    and the first letter appearing after each underscore (underscores are
    removed).

    :param name: name of agent, e.g. local_human

    Returns class of agent, e.g. LocalHumanAgent.
    """
    words = name.split('_')
    class_name = ''
    for w in words:
        class_name += w[0].upper() + w[1:]
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

    :param agent_path: path to model class in one of the above formats.
    """
    repo = 'parlai'
    if agent_path.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        # this will follow the same paths but look in parlai_internal instead
        repo = 'parlai_internal'
        agent_path = agent_path[9:]

    if agent_path.startswith('legacy:'):
        # e.g. -m legacy:seq2seq:0
        # will check legacy_agents.seq2seq.seq2seq_v0:Seq2seqAgent
        s = agent_path.split(':')
        if len(s) != 3:
            raise RuntimeError(
                'legacy paths should follow pattern '
                'legacy:model:version; you used {}'
                ''.format(agent_path)
            )
        model_name = s[1]  # seq2seq
        module_name = 'parlai.agents.legacy_agents.{m}.{m}_v{v}'.format(
            m=model_name, v=s[2]
        )
        class_name = name_to_agent_class(model_name)
    elif agent_path.startswith('projects:'):
        # e.g. -m projects:personachat:kvmemnn
        s = agent_path.split(':')
        if len(s) != 3:
            raise RuntimeError(
                'projects paths should follow pattern '
                'projects:folder:model; you used {}'
                ''.format(agent_path)
            )
        folder_name = s[1]
        model_name = s[2]
        module_name = 'projects.{p}.{m}.{m}'.format(m=model_name, p=folder_name)
        class_name = name_to_agent_class(model_name)
    elif ':' in agent_path:
        # e.g. -m "parlai.agents.seq2seq.seq2seq:Seq2seqAgent"
        s = agent_path.split(':')
        module_name = s[0]
        class_name = s[1]
    elif '/' in agent_path:
        # e.g. -m my_agent/special_variant
        # will check parlai.agents.my_agent.special_variant:SpecialVariantAgent
        sp = agent_path.split('/')
        module_name = "%s.agents.%s.%s" % (repo, sp[0], sp[1])
        class_name = name_to_agent_class(sp[1])
    else:
        # e.g. -m seq2seq
        # will check parlai.agents.seq2seq.agents for Seq2seqAgent first
        # then check parlai.agents.seq2seq.seq2seq for Seq2seqAgent second
        class_name = name_to_agent_class(agent_path)
        try:
            module_name = "%s.agents.%s.agents" % (repo, agent_path)
            importlib.import_module(module_name)  # check if it's there
        except ImportError:
            module_name = "%s.agents.%s.%s" % (repo, agent_path, agent_path)
    my_module = importlib.import_module(module_name)
    model_class = getattr(my_module, class_name)
    return model_class


# TASK LOADER
def load_task_module(taskname: str):
    """
    Get the module of the task agent specified by `--task`.

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
    if len(sp) > 1 and '=' not in sp[1]:
        sp[1] = sp[1][0].upper() + sp[1][1:]
        teacher = sp[1]
        if '.' not in sp[0] and 'Teacher' not in teacher:
            # Reformat from underscore to CamelCase and append "Teacher" to
            # class name by default if a complete path is not given.
            words = teacher.split('_')
            teacher_name = ''
            for w in words:
                teacher_name += w[0].upper() + w[1:]
            teacher = teacher_name + "Teacher"
    else:
        teacher = "DefaultTeacher"
    my_module = importlib.import_module(module_name)
    teacher_class = getattr(my_module, teacher)
    return teacher_class


# WORLD LOADER
def load_world_module(
    taskname: str,
    interactive_task: bool,
    num_agents=None,  # a priori may not know the number of agents
    default_world=None
):
    sp = taskname.strip()
    repo = 'parlai'
    if sp.startswith('internal:'):
        # To switch to local repo, useful for non-public projects
        # (make a directory called 'parlai_internal' with your private agents)
        repo = 'parlai_internal'
        sp = sp[9:]
    sp = sp.split(':')
    if '.' in sp[0]:
        # TODO: clean this logic up, kinda messy
        # The case of opt['task'] = 'parlai.tasks.squad.agents:DefaultTeacher'
        # (i.e. specifying your own path directly, assumes DialogPartnerWorld)
        if default_world is not None:
            world_class = default_world
        elif num_agents is not None:
            import parlai.core.worlds as core_worlds
            world_name = "DialogPartnerWorld" if num_agents == 2 else "MultiAgentDialogWorld"
            world_class = getattr(core_worlds, world_name)
        else:
            return None
    else:
        task = sp[0].lower()
        if len(sp) > 1:
            sp[1] = sp[1][0].upper() + sp[1][1:]
            world_name = sp[1] + "World"
            if interactive_task:
                world_name = "Interactive" + world_name
        else:
            if interactive_task:
                world_name = "InteractiveWorld"
            else:
                world_name = "DefaultWorld"
        module_name = "%s.tasks.%s.worlds" % (repo, task)
        try:
            my_module = importlib.import_module(module_name)
            world_class = getattr(my_module, world_name)
        except (ModuleNotFoundError, AttributeError):
            # Defaults to this if you did not specify a world for your task.
            if default_world is not None:
                world_class = default_world
            elif num_agents is not None:
                import parlai.core.worlds as core_worlds
                world_name = "DialogPartnerWorld" if num_agents == 2 else "MultiAgentDialogWorld"
                world_class = getattr(core_worlds, world_name)
            else:
                world_class = None

    return world_class
