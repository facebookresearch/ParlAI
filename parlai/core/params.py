# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Provides an argument parser and a set of default command line options for
using the ParlAI package.
"""

import argparse
import importlib
import os
import sys
from parlai.core.agents import get_agent_module, get_task_module
from parlai.tasks.tasks import ids_to_tasks

def str2bool(value):
    v = value.lower()
    if v in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2class(value):
    """From import path string, returns the class specified. For example, the
    string 'parlai.agents.drqa.drqa:SimpleDictionaryAgent' returns
    <class 'parlai.agents.drqa.drqa.SimpleDictionaryAgent'>.
    """
    if ':' not in value:
        raise RuntimeError('Use a colon before the name of the class.')
    name = value.split(':')
    module = importlib.import_module(name[0])
    return getattr(module, name[1])

def class2str(value):
    """Inverse of params.str2class()."""
    s = str(value)
    s = s[s.find('\'') + 1 : s.rfind('\'')] # pull out import path
    s = ':'.join(s.rsplit('.', 1)) # replace last period with ':'
    return s


class ParlaiParser(argparse.ArgumentParser):
    """Pseudo-extension of ``argparse`` which sets a number of parameters
    for the ParlAI framework. More options can be added specific to other
    modules by passing this object and calling ``add_arg()`` or
    ``add_argument()`` on it.

    For example, see ``parlai.core.dict.DictionaryAgent.add_cmdline_args``.
    """

    def __init__(self, add_parlai_args=True, add_model_args=False,
                 model_argv=None):
        """Initializes the ParlAI argparser.
        - add_parlai_args (default True) initializes the default arguments for
        ParlAI package, including the data download paths and task arguments.
        - add_model_args (default False) initializes the default arguments for
        loading models, including initializing arguments from that model.
        - model_argv (default None uses sys.argv) specifies the list of
        arguments which includes the model name (e.g. `-m drqa`).
        """
        super().__init__(description='ParlAI parser.')
        self.register('type', 'bool', str2bool)
        self.register('type', 'class', str2class)
        self.parlai_home = (os.path.dirname(os.path.dirname(os.path.dirname(
                            os.path.realpath(__file__)))))
        os.environ['PARLAI_HOME'] = self.parlai_home

        self.add_arg = self.add_argument

        if add_parlai_args:
            self.add_parlai_args(model_argv)
            self.add_image_args()
        if add_model_args:
            self.add_model_args(model_argv)

    def add_parlai_data_path(self, argument_group=None):
        if argument_group is None:
            argument_group = self
        default_data_path = os.path.join(self.parlai_home, 'data')
        argument_group.add_argument(
            '-dp', '--datapath', default=default_data_path,
            help='path to datasets, defaults to {parlai_dir}/data')

    def add_mturk_args(self):
        mturk = self.add_argument_group('Mechanical Turk')
        default_log_path = os.path.join(self.parlai_home, 'logs', 'mturk')
        mturk.add_argument(
            '--mturk-log-path', default=default_log_path,
            help='path to MTurk logs, defaults to {parlai_dir}/logs/mturk')
        mturk.add_argument(
            '-t', '--task',
            help='MTurk task, e.g. "qa_data_collection" or "model_evaluator"')
        mturk.add_argument(
            '-nc', '--num-conversations', default=1, type=int,
            help='number of conversations you want to create for this task')
        mturk.add_argument(
            '--unique', dest='unique_worker', default=False,
            action='store_true',
            help='enforce that no worker can work on your task twice')
        mturk.add_argument(
            '-r', '--reward', default=0.05, type=float,
            help='reward for each worker for finishing the conversation, '
                 'in US dollars')
        mturk.add_argument(
            '--sandbox', dest='is_sandbox', action='store_true',
            help='submit the HITs to MTurk sandbox site')
        mturk.add_argument(
            '--live', dest='is_sandbox', action='store_false',
            help='submit the HITs to MTurk live site')
        mturk.add_argument(
            '--debug', dest='is_debug', action='store_true',
            help='print and log all server interactions and messages')
        mturk.add_argument(
            '--verbose', dest='verbose', action='store_true',
            help='print all messages sent to and from Turkers')
        mturk.add_argument(
            '--log-level', dest='log_level', type=int, default=20,
            help='importance level for what to put into the logs. the lower '
                 'the level the more that gets logged. values are 0-50')
        mturk.add_argument(
            '--count-complete', dest='count_complete',
            default=False, action='store_true',
            help='continue until the requested number of conversations are '
                 'completed rather than attempted')
        mturk.add_argument(
            '--allowed-conversations', dest='allowed_conversations',
            default=0, type=int,
            help='number of concurrent conversations that one mturk worker '
                 'is able to be involved in, 0 is unlimited')

        mturk.set_defaults(is_sandbox=True)
        mturk.set_defaults(is_debug=False)
        mturk.set_defaults(verbose=False)

    def add_parlai_args(self, args=None):
        default_downloads_path = os.path.join(self.parlai_home, 'downloads')
        parlai = self.add_argument_group('Main ParlAI Arguments')
        parlai.add_argument(
            '-t', '--task',
            help='ParlAI task(s), e.g. "babi:Task1" or "babi,cbt"')
        parlai.add_argument(
            '--download-path', default=default_downloads_path,
            help='path for non-data dependencies to store any needed files.'
                 'defaults to {parlai_dir}/downloads')
        parlai.add_argument(
            '-dt', '--datatype', default='train',
            choices=['train', 'train:stream', 'train:ordered',
                'train:ordered:stream', 'train:stream:ordered',
                'valid', 'valid:stream', 'test', 'test:stream'],
            help='choose from: train, train:ordered, valid, test. to stream '
                 'data add ":stream" to any option (e.g., train:stream). '
                 'by default: train is random with replacement, '
                 'valid is ordered, test is ordered.')
        parlai.add_argument(
            '-im', '--image-mode', default='raw', type=str,
            help='image preprocessor to use. default is "raw". set to "none" '
                 'to skip image loading.')
        parlai.add_argument(
            '-nt', '--numthreads', default=1, type=int,
            help='number of threads, e.g. for hogwild')
        parlai.add_argument(
            '-bs', '--batchsize', default=1, type=int,
            help='batch size for minibatch training schemes')
        self.add_parlai_data_path(parlai)
        self.add_task_args(args)

    def add_task_args(self, args):
        # Find which task specified, and add its specific arguments.
        args = sys.argv if args is None else args
        task = None
        for index, item in enumerate(args):
            if item == '-t' or item == '--task':
                task = args[index + 1]
        if task:
            for t in ids_to_tasks(task).split(','):
                agent = get_task_module(t)
                if hasattr(agent, 'add_cmdline_args'):
                    agent.add_cmdline_args(self)

    def add_model_args(self, args=None):
        model_args = self.add_argument_group('ParlAI Model Arguments')
        model_args.add_argument(
            '-m', '--model', default=None,
            help='the model class name, should match parlai/agents/<model>')
        model_args.add_argument(
            '-mf', '--model-file', default=None,
            help='model file name for loading and saving models')
        model_args.add_argument(
            '--dict-class',
            help='the class of the dictionary agent uses')
        # Find which model specified, and add its specific arguments.
        if args is None:
            args = sys.argv
        model = None
        for index, item in enumerate(args):
            if item == '-m' or item == '--model':
                model = args[index + 1]
        if model:
            agent = get_agent_module(model)
            if hasattr(agent, 'add_cmdline_args'):
                agent.add_cmdline_args(self)
            if hasattr(agent, 'dictionary_class'):
                s = class2str(agent.dictionary_class())
                model_args.set_defaults(dict_class=s)

    def add_image_args(self, args=None):
        # Find which image mode specified, add its specific arguments if needed.
        args = sys.argv if args is None else args
        image_mode = None
        for index, item in enumerate(args):
            if item == '-im' or item == '--image-mode':
                image_mode = args[index + 1]
        if image_mode and image_mode != 'none':
            parlai = \
                self.add_argument_group('ParlAI Image Preprocessing Arguments')
            parlai.add_argument('--image-size', type=int, default=256,
                help='')
            parlai.add_argument('--image-cropsize', type=int, default=224,
                help='')

    def parse_args(self, args=None, namespace=None, print_args=True):
        """Parses the provided arguments and returns a dictionary of the
        ``args``. We specifically remove items with ``None`` as values in order
        to support the style ``opt.get(key, default)``, which would otherwise
        return ``None``.
        """
        self.args = super().parse_args(args=args)
        self.opt = vars(self.args)

        # custom post-parsing
        self.opt['parlai_home'] = self.parlai_home

        # set environment variables
        if self.opt.get('download_path'):
            os.environ['PARLAI_DOWNPATH'] = self.opt['download_path']
        if self.opt.get('datapath'):
            os.environ['PARLAI_DATAPATH'] = self.opt['datapath']

        if print_args:
            self.print_args()

        return self.opt

    def print_args(self):
        """Print out all the arguments in this parser."""
        if not self.opt:
            self.parse_args(print_args=False)
        values = {}
        for key, value in self.opt.items():
            values[str(key)] = str(value)
        for group in self._action_groups:
            group_dict = {
                a.dest:getattr(self.args,a.dest,None)
                for a in group._group_actions
            }
            namespace = argparse.Namespace(**group_dict)
            count = 0
            for key in namespace.__dict__:
                if key in values:
                    if count == 0:
                        print('[ ' + group.title + ': ] ')
                    count += 1
                    print('[  ' + key + ': ' + values[key] + ' ]')
