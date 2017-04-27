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


def str2bool(value):
    return value.lower() in ('yes', 'true', 't', '1', 'y')


class ParlaiParser(object):
    """Pseudo-extension of argparse which sets a number of parameters for the
    ParlAI framework. More options can be added specific to other modules by
    passing this object and calling `add_arg` or `add_argument` on it.

    For example, see `parlai.core.dict.DictionaryAgent.add_cmdline_args`
    """

    def __init__(self, add_parlai_args=True, add_model_args=False):
        self.parser = argparse.ArgumentParser(description='ParlAI parser.')
        self.parser.register('type', 'bool', str2bool)
        if add_parlai_args:
            self.add_parlai_args()
        if add_model_args:
            self.add_model_args()

        self.add_arg = self.parser.add_argument
        self.add_argument = self.parser.add_argument
        self.register = self.parser.register

    def add_parlai_args(self):
        parlai_dir = (os.path.dirname(os.path.dirname(os.path.dirname(
                      os.path.realpath(__file__)))))
        default_data_path = parlai_dir + '/data/'
        default_downloads_path = parlai_dir + '/downloads/'

        self.parser.add_argument(
            '-t', '--task',
            help='ParlAI task(s), e.g. "babi:Task1" or "babi,cbt"')
        self.parser.add_argument(
            '-dp', '--datapath', default=default_data_path,
            help='path to datasets, defaults to {parlai_dir}/data')
        self.parser.add_argument(
            '--download-path', default=default_downloads_path,
            help='path for non-data dependencies to store any needed files.' +
                 'defaults to {parlai_dir}/downloads')
        self.parser.add_argument(
            '-dt', '--datatype', default='train',
            choices=['train', 'train:ordered', 'valid', 'test'],
            help='choose from: train, train:ordered, valid, test. ' +
                 'by default: train is random with replacement, ' +
                 'valid is ordered, test is ordered.')
        self.parser.add_argument(
            '-nt', '--numthreads', default=1, type=int,
            help='number of threads, e.g. for hogwild')
        self.parser.add_argument(
            '-bs', '--batchsize', default=1, type=int,
            help='batch size for minibatch training schemes')

    def add_model_args(self):
        self.parser.add_argument(
            '-m', '--model', default='repeat_label',
            help='the model class name, should match parlai/agents/<model>')
        self.parser.add_argument(
            '-mp', '--model_params', default='',
            help='the model parameters, a string that is parsed separately '
            + 'by the model parser after the model class is instantiated')
        self.parser.add_argument(
            '-mf', '--model_file', default='',
            help='model file name for loading and saving models')

    def parse_args(self, args=None, print_args=True):
        """Parses the provided arguments and returns a dictionary of the args.
        We specifically remove items with `None` as values in order to support
        the style `opt.get(key, default)`, which would otherwise return None.
        """
        self.args = self.parser.parse_args(args=args)
        self.opt = {k: v for k, v in vars(self.args).items() if v is not None}
        if print_args:
            self.print_args()
        if 'download_path' in self.opt:
            os.environ['PARLAI_DOWNPATH'] = self.opt['download_path']
        return self.opt

    def print_args(self):
        """Print out all the arguments in this parser."""
        if not self.opt:
            self.parse_args(print_args=False)
        for key, value in self.opt.items():
            print('[' + str(key) + ':' + str(value) + ']')
