#!/usr/bin/env python3

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
import pickle
import sys as _sys
import datetime
from parlai.core.agents import get_agent_module, get_task_module
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.build_data import modelzoo_path
from parlai.core.pytorch_data_teacher import get_dataset_classes


def get_model_name(opt):
    model = opt.get('model', None)
    if model is None:
        # try to get model name from model opt file
        model_file = opt.get('model_file', None)
        if model_file is not None:
            optfile = model_file + '.opt'
            if os.path.isfile(optfile):
                with open(optfile, 'rb') as handle:
                    new_opt = pickle.load(handle)
                    model = new_opt.get('model', None)
    return model


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
    s = s[s.find('\'') + 1:s.rfind('\'')]  # pull out import path
    s = ':'.join(s.rsplit('.', 1))  # replace last period with ':'
    return s


def fix_underscores(args):
    """Converts underscores to hyphens in args.

    For example, converts '--gradient_clip' to '--gradient-clip'.

    :param args: iterable, possibly containing args strings with underscores.
    """
    if args:
        new_args = []
        for a in args:
            if type(a) is str and a.startswith('-'):
                a = a.replace('_', '-')
            new_args.append(a)
        args = new_args
    return args


class ParlaiParser(argparse.ArgumentParser):
    """Pseudo-extension of ``argparse`` which sets a number of parameters
    for the ParlAI framework. More options can be added specific to other
    modules by passing this object and calling ``add_arg()`` or
    ``add_argument()`` on it.

    For example, see ``parlai.core.dict.DictionaryAgent.add_cmdline_args``.
    """

    def __init__(
        self,
        add_parlai_args=True,
        add_model_args=False,
        description='ParlAI parser',
    ):
        """Initializes the ParlAI argparser.
        - add_parlai_args (default True) initializes the default arguments for
        ParlAI package, including the data download paths and task arguments.
        - add_model_args (default False) initializes the default arguments for
        loading models, including initializing arguments from that model.
        """
        super().__init__(description=description, allow_abbrev=False,
                         conflict_handler='resolve')
        self.register('type', 'bool', str2bool)
        self.register('type', 'class', str2class)
        self.parlai_home = (os.path.dirname(os.path.dirname(os.path.dirname(
                            os.path.realpath(__file__)))))
        os.environ['PARLAI_HOME'] = self.parlai_home

        self.add_arg = self.add_argument

        # remember which args were specified on the command line
        self.cli_args = _sys.argv[1:]
        self.overridable = {}

        if add_parlai_args:
            self.add_parlai_args()
        if add_model_args:
            self.add_model_args()

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
            '--max-hits-per-worker', dest='max_hits_per_worker', default=0, type=int,
            help='Max number of hits each worker can perform during current group run')
        mturk.add_argument(
            '--unique-qual-name', dest='unique_qual_name',
            default=None, type=str,
            help='qualification name to use for uniqueness between HITs')
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
            '--hard-block', dest='hard_block', action='store_true',
            default=False,
            help='Hard block disconnecting Turkers from all of your HITs')
        mturk.add_argument(
            '--log-level', dest='log_level', type=int, default=20,
            help='importance level for what to put into the logs. the lower '
                 'the level the more that gets logged. values are 0-50')
        mturk.add_argument(
            '--disconnect-qualification', dest='disconnect_qualification',
            default=None,
            help='Qualification to use for soft blocking users for '
                 'disconnects. By default '
                 'turkers are never blocked, though setting this will allow '
                 'you to filter out turkers that have disconnected too many '
                 'times on previous HITs where this qualification was set.')
        mturk.add_argument(
            '--block-qualification', dest='block_qualification', default=None,
            help='Qualification to use for soft blocking users. This '
                 'qualification is granted whenever soft_block_worker is '
                 'called, and can thus be used to filter workers out from a '
                 'single task or group of tasks by noted performance.')
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
        mturk.add_argument(
            '--max-connections', dest='max_connections',
            default=30, type=int,
            help='number of HITs that can be launched at the same time, 0 is '
                 'unlimited.'
        )
        mturk.add_argument(
            '--min-messages', dest='min_messages',
            default=0, type=int,
            help='number of messages required to be sent by MTurk agent when '
                 'considering whether to approve a HIT in the event of a '
                 'partner disconnect. I.e. if the number of messages '
                 'exceeds this number, the turker can submit the HIT.'
        )
        mturk.add_argument(
            '--local', dest='local', default=False, action='store_true',
            help='Run the server locally on this server rather than setting up'
                 ' a heroku server.'
        )
        mturk.add_argument(
            '--hobby', dest='hobby', default=False, action='store_true',
            help='Run the heroku server on the hobby tier.'
        )
        mturk.add_argument(
            '--max-time', dest='max_time', default=0, type=int,
            help='Maximum number of seconds per day that a worker is allowed '
                 'to work on this assignment'
        )
        mturk.add_argument(
            '--max-time-qual', dest='max_time_qual', default=None,
            help='Qualification to use to share the maximum time requirement '
                 'with other runs from other machines.'
        )
        mturk.add_argument(
            '--heroku-team', dest='heroku_team', default=None,
            help='Specify Heroku team name to use for launching Dynos.'
        )

        mturk.set_defaults(is_sandbox=True)
        mturk.set_defaults(is_debug=False)
        mturk.set_defaults(verbose=False)

    def add_messenger_args(self):
        messenger = self.add_argument_group('Facebook Messenger')
        messenger.add_argument(
            '--debug', dest='is_debug', action='store_true',
            help='print and log all server interactions and messages')
        messenger.add_argument(
            '--verbose', dest='verbose', action='store_true',
            help='print all messages sent to and from Turkers')
        messenger.add_argument(
            '--log-level', dest='log_level', type=int, default=20,
            help='importance level for what to put into the logs. the lower '
                 'the level the more that gets logged. values are 0-50')
        messenger.add_argument(
            '--force-page-token', dest='force_page_token', action='store_true',
            help='override the page token stored in the cache for a new one')
        messenger.add_argument(
            '--password', dest='password', type=str, default=None,
            help='Require a password for entry to the bot')
        messenger.add_argument(
            '--bypass-server-setup', dest='bypass_server_setup',
            action='store_true', default=False,
            help='should bypass traditional server and socket setup')
        messenger.add_argument(
            '--local', dest='local', action='store_true', default=False,
            help='Run the server locally on this server rather than setting up'
                 ' a heroku server.'
        )

        messenger.set_defaults(is_debug=False)
        messenger.set_defaults(verbose=False)

    def add_parlai_args(self, args=None):
        default_downloads_path = os.path.join(self.parlai_home, 'downloads')
        parlai = self.add_argument_group('Main ParlAI Arguments')
        parlai.add_argument(
            '-t', '--task',
            help='ParlAI task(s), e.g. "babi:Task1" or "babi,cbt"')
        parlai.add_argument(
            '-pyt', '--pytorch-teacher-task',
            help='Specify to use the PytorchDataTeacher for multiprocessed '
                 'data loading with a standard ParlAI task, e.g. "babi:Task1k"'
        )
        parlai.add_argument(
            '-pytd', '--pytorch-teacher-dataset',
            help='Specify to use the PytorchDataTeacher for multiprocessed '
                 'data loading with a pytorch Dataset, e.g. "vqa_1" or "flickr30k"'
        )
        parlai.add_argument(
            '--download-path', default=default_downloads_path,
            help='path for non-data dependencies to store any needed files.'
                 'defaults to {parlai_dir}/downloads')
        parlai.add_argument(
            '-dt', '--datatype', default='train',
            choices=[
                'train',
                'train:stream',
                'train:ordered',
                'train:ordered:stream',
                'train:stream:ordered',
                'train:evalmode',
                'train:evalmode:stream',
                'train:evalmode:ordered',
                'train:evalmode:ordered:stream',
                'train:evalmode:stream:ordered',
                'valid',
                'valid:stream',
                'test',
                'test:stream'
            ],
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
            help='number of threads. If batchsize set to 1, used for hogwild; '
                 'otherwise, used for number of threads in threadpool loading,'
                 ' e.g. in vqa')
        parlai.add_argument(
            '--hide-labels', default=False, type='bool',
            help='default (False) moves labels in valid and test sets to the '
                 'eval_labels field. If True, they are hidden completely.')
        batch = self.add_argument_group('Batching Arguments')
        batch.add_argument(
            '-bs', '--batchsize', default=1, type=int,
            help='batch size for minibatch training schemes')
        batch.add_argument('-bsrt', '--batch-sort', default=False, type='bool',
                           help='If enabled (default %(default)s), create batches by '
                                'flattening all episodes to have exactly one '
                                'utterance exchange and then sorting all the '
                                'examples according to their length. This '
                                'dramatically reduces the amount of padding '
                                'present after examples have been parsed, '
                                'speeding up training.')
        batch.add_argument('-clen', '--context-length', default=-1, type=int,
                           help='Number of past utterances to remember when '
                                'building flattened batches of data in multi-'
                                'example episodes.')
        batch.add_argument('-incl', '--include-labels',
                           default=True, type='bool',
                           help='Specifies whether or not to include labels '
                                'as past utterances when building flattened '
                                'batches of data in multi-example episodes.')
        pytorch = self.add_argument_group('PytorchData Arguments')
        pytorch.add_argument(
            '--pytorch-datafile', type=str, default=None,
            help='datafile for pytorch data loader')
        pytorch.add_argument(
            '-nw', '--numworkers', type=int, default=4,
            help='how many workers the Pytorch dataloader should use')
        pytorch.add_argument(
            '--pytorch-preprocess', type='bool', default=False,
            help='Whether the agent should preprocess the data while building'
                 'the pytorch data')
        pytorch.add_argument(
            '--batch-sort-cache', type=str,
            choices=['pop', 'index', 'none'], default='none',
            help='Whether to have batches of similarly sized episodes, and how'
            'to build up the cache')
        pytorch.add_argument(
            '--batch-length-range', type=int, default=5,
            help='degree of variation of size allowed in batch')
        pytorch.add_argument(
            '--shuffle', type='bool', default=False,
            help='Whether to shuffle the data')
        self.add_parlai_data_path(parlai)

    def add_model_args(self):
        """Add arguments related to models such as model files."""
        model_args = self.add_argument_group('ParlAI Model Arguments')
        model_args.add_argument(
            '-m', '--model', default=None,
            help='the model class name. can match parlai/agents/<model> for '
                 'agents in that directory, or can provide a fully specified '
                 'module for `from X import Y` via `-m X:Y` '
                 '(e.g. `-m parlai.agents.seq2seq.seq2seq:Seq2SeqAgent`)')
        model_args.add_argument(
            '-mf', '--model-file', default=None,
            help='model file name for loading and saving models')
        model_args.add_argument(
            '--dict-class',
            help='the class of the dictionary agent uses')

    def add_model_subargs(self, model):
        """Add arguments specific to a particular model."""
        agent = get_agent_module(model)
        try:
            if hasattr(agent, 'add_cmdline_args'):
                agent.add_cmdline_args(self)
        except argparse.ArgumentError:
            # already added
            pass
        try:
            if hasattr(agent, 'dictionary_class'):
                s = class2str(agent.dictionary_class())
                self.set_defaults(dict_class=s)
        except argparse.ArgumentError:
            # already added
            pass

    def add_task_args(self, task):
        """Add arguments specific to the specified task."""
        for t in ids_to_tasks(task).split(','):
            agent = get_task_module(t)
            try:
                if hasattr(agent, 'add_cmdline_args'):
                    agent.add_cmdline_args(self)
            except argparse.ArgumentError:
                # already added
                pass

    def add_pyt_dataset_args(self, opt):
        """Add arguments specific to specified pytorch dataset"""
        dataset_classes = get_dataset_classes(opt)
        for dataset, _, _ in dataset_classes:
            try:
                if hasattr(dataset, 'add_cmdline_args'):
                    dataset.add_cmdline_args(self)
            except argparse.ArgumentError:
                # already added
                pass

    def add_image_args(self, image_mode):
        """Add additional arguments for handling images."""
        try:
            parlai = self.add_argument_group('ParlAI Image Preprocessing Arguments')
            parlai.add_argument('--image-size', type=int, default=256,
                                help='resizing dimension for images')
            parlai.add_argument('--image-cropsize', type=int, default=224,
                                help='crop dimension for images')
        except argparse.ArgumentError:
            # already added
            pass

    def add_extra_args(self, args=None):
        """Add more args depending on how known args are set."""
        parsed = vars(self.parse_known_args(args, nohelp=True)[0])

        # find which image mode specified if any, and add additional arguments
        image_mode = parsed.get('image_mode', None)
        if image_mode is not None and image_mode != 'none':
            self.add_image_args(image_mode)

        # find which task specified if any, and add its specific arguments
        task = parsed.get('task', None)
        if task is not None:
            self.add_task_args(task)
        evaltask = parsed.get('evaltask', None)
        if evaltask is not None:
            self.add_task_args(evaltask)

        # find pytorch teacher task if specified, add its specific arguments
        pytorch_teacher_task = parsed.get('pytorch_teacher_task', None)
        if pytorch_teacher_task is not None:
            self.add_task_args(pytorch_teacher_task)

        # find pytorch dataset if specified, add its specific arguments
        pytorch_teacher_dataset = parsed.get('pytorch_teacher_dataset', None)
        if pytorch_teacher_dataset is not None:
            self.add_pyt_dataset_args(parsed)

        # find which model specified if any, and add its specific arguments
        model = get_model_name(parsed)
        if model is not None:
            self.add_model_subargs(model)

        # reset parser-level defaults over any model-level defaults
        try:
            self.set_defaults(**self._defaults)
        except AttributeError:
            raise RuntimeError('Please file an issue on github that argparse '
                               'got an attribute error when parsing.')

    def parse_known_args(self, args=None, namespace=None, nohelp=False):
        """Custom parse known args to ignore help flag."""
        if args is None:
            # args default to the system args
            args = _sys.argv[1:]
        args = fix_underscores(args)

        if nohelp:
            # ignore help
            args = [a for a in args if a != '-h' and a != '--help']
        return super().parse_known_args(args, namespace)

    def parse_args(self, args=None, namespace=None, print_args=True):
        """Parses the provided arguments and returns a dictionary of the
        ``args``. We specifically remove items with ``None`` as values in order
        to support the style ``opt.get(key, default)``, which would otherwise
        return ``None``.
        """
        self.add_extra_args(args)
        self.args = super().parse_args(args=args)
        self.opt = vars(self.args)

        # custom post-parsing
        self.opt['parlai_home'] = self.parlai_home
        if 'batchsize' in self.opt and self.opt['batchsize'] <= 1:
            # hide batch options
            self.opt.pop('batch_sort', None)
            self.opt.pop('context_length', None)

        # set environment variables
        if self.opt.get('download_path'):
            os.environ['PARLAI_DOWNPATH'] = self.opt['download_path']
        if self.opt.get('datapath'):
            os.environ['PARLAI_DATAPATH'] = self.opt['datapath']

        # set all arguments specified in commandline as overridable
        option_strings_dict = {}
        store_true = []
        store_false = []
        for group in self._action_groups:
            for a in group._group_actions:
                if hasattr(a, 'option_strings'):
                    for option in a.option_strings:
                        option_strings_dict[option] = a.dest
                        if '_StoreTrueAction' in str(type(a)):
                            store_true.append(option)
                        elif '_StoreFalseAction' in str(type(a)):
                            store_false.append(option)

        for i in range(len(self.cli_args)):
            if self.cli_args[i] in option_strings_dict:
                if self.cli_args[i] in store_true:
                    self.overridable[option_strings_dict[self.cli_args[i]]] = \
                        True
                elif self.cli_args[i] in store_false:
                    self.overridable[option_strings_dict[self.cli_args[i]]] = \
                        False
                elif i < len(self.cli_args) - 1 and self.cli_args[i + 1][:1] != '-':
                    key = option_strings_dict[self.cli_args[i]]
                    self.overridable[key] = self.opt[key]
        self.opt['override'] = self.overridable

        # map filenames that start with 'models:' to point to the model zoo dir
        if self.opt.get('model_file') is not None:
            self.opt['model_file'] = modelzoo_path(self.opt.get('datapath'),
                                                   self.opt['model_file'])
        if self.opt['override'].get('model_file') is not None:
            # also check override
            self.opt['override']['model_file'] = modelzoo_path(
                self.opt.get('datapath'), self.opt['override']['model_file'])
        if self.opt.get('dict_file') is not None:
            self.opt['dict_file'] = modelzoo_path(self.opt.get('datapath'),
                                                  self.opt['dict_file'])
        if self.opt['override'].get('dict_file') is not None:
            # also check override
            self.opt['override']['dict_file'] = modelzoo_path(
                self.opt.get('datapath'), self.opt['override']['dict_file'])

        # add start time of an experiment
        self.opt['starttime'] = datetime.datetime.today().strftime('%b%d_%H-%M')

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
                a.dest: getattr(self.args, a.dest, None)
                for a in group._group_actions
            }
            namespace = argparse.Namespace(**group_dict)
            count = 0
            for key in sorted(namespace.__dict__):
                if key in values:
                    if count == 0:
                        print('[ ' + group.title + ': ] ')
                    count += 1
                    print('[  ' + key + ': ' + values[key] + ' ]')

    def set_params(self, **kwargs):
        """Set overridable kwargs."""
        self.set_defaults(**kwargs)
        for k, v in kwargs.items():
            self.overridable[k] = v

    def add_argument(self, *args, **kwargs):
        """Override to convert underscores to hyphens for consistency."""
        return super().add_argument(*fix_underscores(args), **kwargs)

    def add_argument_group(self, *args, **kwargs):
        """Override to make arg groups also convert underscores to hyphens."""
        arg_group = super().add_argument_group(*args, **kwargs)
        original_add_arg = arg_group.add_argument

        def ag_add_argument(*args, **kwargs):
            return original_add_arg(*fix_underscores(args), **kwargs)

        arg_group.add_argument = ag_add_argument  # override _ => -
        return arg_group
