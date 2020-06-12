#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Common Abstract classes for many agents.

This module provides a set of basic agents:

    ``Agent(object)``
    base class for all other agents, implements the ``observe()`` method
    which receives an observation/action dict and the ``act()`` method which
    returns a dict in response.

    ``Teacher(Agent)``
    also implements the ``report()`` method for returning metrics. All ParlAI
    tasks implement the ``Teacher`` class.

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
from parlai.core.loader import load_agent_module
from parlai.core.loader import register_agent  # noqa: F401
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
import copy
import os
import parlai.utils.logging as logging


class Agent(object):
    """
    Base class for all other agents.
    """

    def __init__(self, opt: Opt, shared=None):
        if not hasattr(self, 'id'):
            self.id = 'agent'
        if not hasattr(self, 'opt'):
            self.opt = copy.deepcopy(opt)
        self.observation = None

    def observe(self, observation):
        """
        Receive an observation/action dict.
        """
        self.observation = observation
        return observation

    def act(self):
        """
        Return an observation/action dict based upon given observation.
        """
        if hasattr(self, 'observation') and self.observation is not None:
            logging.info(f'agent received observation:\n{self.observation}')

        t = {}
        t['text'] = 'hello, teacher!'
        logging.info(f'agent sending message:\n{t}')
        return t

    def getID(self):
        """
        Return the agent ID.
        """
        return self.id

    def epoch_done(self):
        """
        Return whether the epoch is done or not.

        :rtype: boolean
        """
        return False

    def reset(self):
        """
        Reset the agent, clearing its observation.

        Many subclasses implement additional reset logic.
        """
        self.observation = None

    def reset_metrics(self):
        """
        Reset any metrics reported by this agent.

        This is called to indicate metrics should start fresh, and is typically called
        between loggings or after a `report()`.
        """
        pass

    def save(self, path=None):
        """
        Save any parameters needed to recreate this agent from loaded parameters.

        Default implementation is no-op, but many subagents implement this logic.
        """
        pass

    def clone(self):
        """
        Make a shared copy of this agent.

        Should be the same as using create_agent_from_shared(.), but slightly easier.
        """
        return type(self)(self.opt, self.share())

    def share(self):
        """
        Share any parameters needed to create a shared version of this agent.

        Default implementation shares the class and the opt, but most agents will want
        to also add model weights, teacher data, etc. This especially useful for
        avoiding providing pointers to large objects to all agents in a batch.
        """
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        return shared

    def shutdown(self):
        """
        Perform any final cleanup if needed.
        """
        pass

    @classmethod
    def upgrade_opt(cls, opt_from_disk: Opt):
        """
        Upgrade legacy options when loading an opt file from disk.

        This is primarily made available to provide a safe space to handle
        backwards-compatible behavior. For example, perhaps we introduce a
        new option today, which wasn't previously available. We can have the
        argument have a new default, but fall back to the "legacy" compatibility
        behavior if the option doesn't exist.

        ``upgrade_opt`` provides an opportunity for such checks for backwards
        compatibility. It is called shortly after loading the opt file from
        disk, and is called before the Agent is initialized.

        Other possible examples include:

            1. Renaming an option,
            2. Deprecating an old option,
            3. Splitting coupled behavior, etc.

        Implementations of ``upgrade_opt`` should conform to high standards,
        due to the risk of these methods becoming complicated and difficult to
        reason about. We recommend the following behaviors:

            1. ``upgrade_opt`` should only be used to provide backwards
            compatibility.  Other behavior should find a different location.
            2. Children should always call the parent's ``upgrade_opt`` first.
            3. ``upgrade_opt`` should always warn when an option was overwritten.
            4. Include comments annotating the date and purpose of each upgrade.
            5. Add an integration test which ensures your old work behaves
            appropriately.

        :param Opt opt_from_disk:
            The opt file, as loaded from the ``.opt`` file on disk.
        :return:
            The modified options
        :rtype:
            Opt
        """
        # 2019-07-11: currently a no-op.
        return opt_from_disk


def compare_init_model_opts(opt: Opt, curr_opt: Opt):
    """
    Print loud warning when `init_model` opts differ from previous configuration.
    """
    if opt.get('init_model') is None:
        return
    opt['init_model'] = modelzoo_path(opt['datapath'], opt['init_model'])
    optfile = opt['init_model'] + '.opt'
    if not os.path.isfile(optfile):
        return
    init_model_opt = Opt.load(optfile)

    extra_opts = {}
    different_opts = {}
    exempt_opts = [
        'model_file',
        'dict_file',
        'override',
        'starttime',
        'init_model',
        'batchindex',
    ]

    # search through init model opts
    for k, v in init_model_opt.items():
        if (
            k not in exempt_opts
            and k in init_model_opt
            and init_model_opt[k] != curr_opt.get(k)
        ):
            if isinstance(v, list):
                if init_model_opt[k] != list(curr_opt[k]):
                    different_opts[k] = ','.join([str(x) for x in v])
            else:
                different_opts[k] = v

    # search through opts to load
    for k, v in curr_opt.items():
        if k not in exempt_opts and k not in init_model_opt:
            if isinstance(v, list):
                extra_opts[k] = ','.join([str(x) for x in v])
            else:
                extra_opts[k] = v

    # print warnings
    extra_strs = ['{}: {}'.format(k, v) for k, v in extra_opts.items()]
    if extra_strs:
        logging.warn(
            'your model is being loaded with opts that do not '
            'exist in the model you are initializing the weights with: '
            '{}'.format(','.join(extra_strs))
        )

    different_strs = [
        '--{} {}'.format(k, v).replace('_', '-') for k, v in different_opts.items()
    ]
    if different_strs:
        logging.warn(
            'your model is being loaded with opts that differ '
            'from the model you are initializing the weights with. Add the '
            'following args to your run command to change this: \n'
            '{}'.format(' '.join(different_strs))
        )


def create_agent_from_model_file(model_file, opt_overides=None):
    """
    Load agent from model file if it exists.

    :param opt_overrides:
        An optional dict of option overrides can also be provided.
    :return:
        The agent
    """
    opt = {}
    opt['model_file'] = model_file
    if opt_overides is None:
        opt_overides = {}
    opt['override'] = opt_overides
    return create_agent_from_opt_file(opt)


def create_agent_from_opt_file(opt: Opt):
    """
    Load agent options and module from file if opt file exists.

    Checks to see if file exists opt['model_file'] + ".opt"; if so, load up the
    options from the file and use that to create an agent, loading the model
    type from that file and overriding any options specified in that file when
    instantiating the agent.

    If that file does not exist, return None.
    """
    model_file = opt['model_file']
    optfile = model_file + '.opt'
    if os.path.isfile(optfile):
        new_opt = Opt.load(optfile)
        # TODO we need a better way to say these options are never copied...
        if 'datapath' in new_opt:
            # never use the datapath from an opt dump
            del new_opt['datapath']
        if 'batchindex' in new_opt:
            # This saved variable can cause trouble if we switch to BS=1 at test time
            del new_opt['batchindex']
        # only override opts specified in 'override' dict
        if opt.get('override'):
            for k, v in opt['override'].items():
                if k in new_opt and str(v) != str(new_opt.get(k)):
                    logging.warn(
                        f"overriding opt['{k}'] to {v} (previously: {new_opt.get(k)})"
                    )
                new_opt[k] = v

        model_class = load_agent_module(new_opt['model'])

        # check for model version
        if hasattr(model_class, 'model_version'):
            curr_version = new_opt.get('model_version', 0)
            if curr_version != model_class.model_version():
                model = new_opt['model']
                m = (
                    'It looks like you are trying to load an older version of'
                    ' the selected model. Change your model argument to use '
                    'the old version from parlai/agents/legacy_agents: for '
                    'example: `-m legacy:{m}:{v}` or '
                    '`--model parlai.agents.legacy_agents.{m}.{m}_v{v}:{c}`'
                )
                if '.' not in model:
                    # give specific error message if it's easy
                    raise RuntimeError(
                        m.format(m=model, v=curr_version, c=model_class.__name__)
                    )
                else:
                    # otherwise generic one
                    raise RuntimeError(
                        m.format(m='modelname', v=curr_version, c='ModelAgent')
                    )

        if hasattr(model_class, 'upgrade_opt'):
            new_opt = model_class.upgrade_opt(new_opt)

        # add model arguments to new_opt if they aren't in new_opt already
        for k, v in opt.items():
            if k not in new_opt:
                new_opt[k] = v
        new_opt['model_file'] = model_file
        if not new_opt.get('dict_file'):
            new_opt['dict_file'] = model_file + '.dict'
        elif new_opt.get('dict_file') and not os.path.isfile(new_opt['dict_file']):
            old_dict_file = new_opt['dict_file']
            new_opt['dict_file'] = model_file + '.dict'
        if not os.path.isfile(new_opt['dict_file']):
            warn_once(
                'WARNING: Neither the specified dict file ({}) nor the '
                '`model_file`.dict file ({}) exists, check to make sure either '
                'is correct. This may manifest as a shape mismatch later '
                'on.'.format(old_dict_file, new_opt['dict_file'])
            )

        # if we want to load weights from --init-model, compare opts with
        # loaded ones
        compare_init_model_opts(opt, new_opt)
        return model_class(new_opt)
    else:
        return None


def create_agent(opt: Opt, requireModelExists=False):
    """
    Create an agent from the options ``model``, ``model_params`` and ``model_file``.

    The input is either of the form
    ``parlai.agents.ir_baseline.agents:IrBaselineAgent`` (i.e. the path
    followed by the class name) or else just ``ir_baseline`` which
    assumes the path above, and a class name suffixed with 'Agent'.

    If ``model-file`` is available in the options this function can also
    attempt to load the model from that location instead. This avoids having to
    specify all the other options necessary to set up the model including its
    name as they are all loaded from the options file if it exists (the file
    opt['model_file'] + '.opt' must exist and contain a pickled or json dict
    containing the model's options).
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
            raise RuntimeError(
                'WARNING: Model file does not exist, check to make '
                'sure it is correct: {}'.format(opt['model_file'])
            )
        # Attempt to load the model from the model file first (this way we do
        # not even have to specify the model name as a parameter)
        model = create_agent_from_opt_file(opt)
        if model is not None:
            return model
        else:
            logging.info(f"No model with opt yet at: {opt['model_file']}(.opt)")

    if opt.get('model'):
        model_class = load_agent_module(opt['model'])
        # if we want to load weights from --init-model, compare opts with
        # loaded ones
        compare_init_model_opts(opt, opt)
        model = model_class(opt)
        if requireModelExists and hasattr(model, 'load') and not opt.get('model_file'):
            # double check that we didn't forget to set model_file on loadable model
            logging.warn('model_file unset but model has a `load` function.')
        return model
    else:
        raise RuntimeError('Need to set `model` argument to use create_agent.')


# Helper functions to create agent/agents given shared parameters
# returned from agent.share(). Useful for parallelism, sharing params, etc.
def create_agent_from_shared(shared_agent):
    """
    Instantiate an agent from the default `shared` params.

    :param shared_agent:
        should include an `opt` dictionary and agent `class`, along with
        whatever other parameters the agent needs to instantiate.
    """
    opt = copy.deepcopy(shared_agent['opt'])
    a = shared_agent['class'](opt, shared_agent)
    return a


def create_agents_from_shared(shared):
    """
    Create agents based on shared data.

    :param shared: `list` of `dict` objects created by calling e.g.
        [a.share() for a in agents].

    Returns a list of instantiated agents.
    """
    shared_agents = []
    for shared_agent in shared:
        agent = create_agent_from_shared(shared_agent)
        shared_agents.append(agent)
    return shared_agents
