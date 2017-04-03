#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
"""Provides an argument parser and a set of default command line options for
using the ParlAI package.
"""

import argparse
import os


class ParlaiParser(object):
    """Pseudo-extension of argparse which sets a number of parameters for the
    ParlAI framework. More options can be added specific to other modules by
    passing this object and calling `add_arg` or `add_argument` on it.

    For example, see `parlai.core.dict.DictionaryAgent.add_cmdline_args`
    """

    def __init__(self, add_parlai_args=True):
        self.parser = argparse.ArgumentParser(description='ParlAI parser.')
        if add_parlai_args:
            self.add_parlai_args()

        self.add_arg = self.parser.add_argument
        self.add_argument = self.parser.add_argument

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
            help='path to datasets, defaults to path pointed to' +
                 ' in parlai/core/data_path.py')
        self.parser.add_argument(
            '--download-path', default=default_downloads_path,
            help='path for non-data dependencies to store any needed files')
        self.parser.add_argument(
            '-dt', '--datatype', default='train',
            choices=['train', 'train:ordered', 'valid', 'test'],
            help='choose from: train, train:ordered, valid, test. ' +
                 'by default: train is random with replacement, ' +
                 'valid is ordered, test is ordered.')
        self.parser.add_argument(
            '-n', '--numthreads', default=1, type=int,
            help='number of threads, e.g. for hogwild')

    def parse_args(self, print_args=True):
        """Parses the provided arguments and returns a dictionary of the args.
        We specifically remove items with `None` as values in order to support
        the style `opt.get(key, default)`, which would otherwise return None.
        """
        self.args = self.parser.parse_args()
        self.opt = {k: v for k, v in vars(self.args).items() if v is not None}
        if print_args:
            self.print_args()
        os.environ['PARLAI_DOWNPATH'] = self.opt['download_path']
        return self.opt

    def print_args(self):
        """Print out all the arguments in this parser."""
        if not self.opt:
            self.parse_args(False)
        for key, value in self.opt.items():
            print('[' + str(key) + ':' + str(value) + ']')
