# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Provides an argument parser and a set of default command line options for
using the ParlAI package.
"""

import argparse
import os
import sys
from parlai.core.agents import get_agent_module

def str2bool(value):
    return value.lower() in ('yes', 'true', 't', '1', 'y')


class ParlaiParser(argparse.ArgumentParser):
    """Pseudo-extension of ``argparse`` which sets a number of parameters for the
    ParlAI framework. More options can be added specific to other modules by
    passing this object and calling ``add_arg()`` or ``add_argument()`` on it.

    For example, see ``parlai.core.dict.DictionaryAgent.add_cmdline_args``.
    """

    def __init__(self, add_parlai_args=True, add_model_args=False, model_argv=None):
        """Initializes the ParlAI argparser.
        - add_parlai_args (default True) initializes the default arguments for the
        ParlAI package, including the data download paths and task arguments.
        - add_model_args (default False) initializes the default arguments for
        loading models, including initializing arguments from that model.
        - model_argv (default None uses sys.argv) specifies the list of
        arguments which includes the model name (e.g. `-m drqa`).
        """
        super().__init__(description='ParlAI parser.')
        self.register('type', 'bool', str2bool)
        self.parlai_home = (os.path.dirname(os.path.dirname(os.path.dirname(
                            os.path.realpath(__file__)))))
        os.environ['PARLAI_HOME'] = self.parlai_home

        self.add_arg = self.add_argument

        if add_parlai_args:
            self.add_parlai_args()
        if add_model_args:
            self.add_model_args(model_argv)

    def add_parlai_data_path(self):
        default_data_path = os.path.join(self.parlai_home, 'data')
        self.add_argument(
            '-dp', '--datapath', default=default_data_path,
            help='path to datasets, defaults to {parlai_dir}/data')

    def add_mturk_args(self):
        default_log_path = os.path.join(self.parlai_home, 'logs', 'mturk')
        self.add_argument(
            '--mturk-log-path', default=default_log_path,
            help='path to MTurk logs, defaults to {parlai_dir}/logs/mturk')
        self.add_argument(
            '-t', '--task',
            help='MTurk task, e.g. "qa_data_collection" or "model_evaluator"')
        self.add_argument(
            '-nh', '--num-hits', default=2, type=int,
            help='number of HITs you want to create for this task')
        self.add_argument(
            '-na', '--num-assignments', default=1, type=int,
            help='number of assignments for each HIT')
        self.add_argument(
            '-r', '--reward', default=0.05, type=float,
            help='reward for each HIT, in US dollars')
        self.add_argument(
            '--sandbox', dest='is_sandbox', action='store_true',
            help='submit the HITs to MTurk sandbox site')
        self.add_argument(
            '--live', dest='is_sandbox', action='store_false',
            help='submit the HITs to MTurk live site')
        self.set_defaults(is_sandbox=True)
        self.add_argument(
            '--verbose', dest='verbose', action='store_true',
            help='print out all messages sent/received in all conversations')
        self.set_defaults(verbose=False)

    def add_parlai_args(self):
        default_downloads_path = os.path.join(self.parlai_home, 'downloads')
        self.add_argument(
            '-t', '--task',
            help='ParlAI task(s), e.g. "babi:Task1" or "babi,cbt"')
        self.add_argument(
            '--download-path', default=default_downloads_path,
            help='path for non-data dependencies to store any needed files.' +
                 'defaults to {parlai_dir}/downloads')
        self.add_argument(
            '-dt', '--datatype', default='train',
            choices=['train', 'train:ordered', 'valid', 'test'],
            help='choose from: train, train:ordered, valid, test. ' +
                 'by default: train is random with replacement, ' +
                 'valid is ordered, test is ordered.')
        self.add_argument(
            '-im', '--image-mode', default='raw', type=str,
            help='image preprocessor to use. default is "raw". set to "none" '
                 'to skip image loading.')
        self.add_argument(
            '-nt', '--numthreads', default=1, type=int,
            help='number of threads, e.g. for hogwild')
        self.add_argument(
            '-bs', '--batchsize', default=1, type=int,
            help='batch size for minibatch training schemes')
        self.add_parlai_data_path()

    def add_model_args(self, args=None):
        self.add_argument(
            '-m', '--model', default='repeat_label',
            help='the model class name, should match parlai/agents/<model>')
        self.add_argument(
            '-mf', '--model-file', default='',
            help='model file name for loading and saving models')
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
                self.add_argument(
                    '--dict-class', default=agent.dictionary_class(), type=str,
                    help='the class of the dictionary agent used')

    def parse_args(self, args=None, namespace=None, print_args=True):
        """Parses the provided arguments and returns a dictionary of the ``args``.
        We specifically remove items with ``None`` as values in order to support
        the style ``opt.get(key, default)``, which would otherwise return ``None``.
        """
        self.args = super().parse_args(args=args)
        self.opt = {k: v for k, v in vars(self.args).items() if v is not None}
        self.opt['parlai_home'] = self.parlai_home
        if 'download_path' in self.opt:
            self.opt['download_path'] = self.opt['download_path']
            os.environ['PARLAI_DOWNPATH'] = self.opt['download_path']
        if 'datapath' in self.opt:
            self.opt['datapath'] = self.opt['datapath']
        if print_args:
            self.print_args()
        return self.opt

    def print_args(self):
        """Print out all the arguments in this parser."""
        if not self.opt:
            self.parse_args(print_args=False)
        for key, value in self.opt.items():
            print('[' + str(key) + ':' + str(value) + ']')
