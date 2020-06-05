#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Provide an argument parser and default command line options for using ParlAI.
"""

import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai

try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    # silence the error
    GIT_AVAILABLE = False

import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import load_teacher_module, load_agent_module, load_world_module
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt

from typing import List, Optional


def print_git_commit():
    """
    Print the current git commit of ParlAI and parlai_internal.
    """
    root = os.path.dirname(os.path.dirname(parlai.__file__))
    internal_root = os.path.join(root, 'parlai_internal')
    try:
        git_ = git.Git(root)
        current_commit = git_.rev_parse('HEAD')
        logging.info(f'Current ParlAI commit: {current_commit}')
    except git.GitCommandNotFound:
        pass
    except git.GitCommandError:
        pass

    try:
        git_ = git.Git(internal_root)
        internal_commit = git_.rev_parse('HEAD')
        logging.info(f'Current internal commit: {internal_commit}')
    except git.GitCommandNotFound:
        pass
    except git.GitCommandError:
        pass


def print_announcements(opt):
    """
    Output any announcements the ParlAI team wishes to make to users.

    Also gives the user the option to suppress the output.
    """
    # no annoucements to make right now
    return

    noannounce_file = os.path.join(opt.get('datapath'), 'noannouncements')
    if os.path.exists(noannounce_file):
        # user has suppressed announcements, don't do anything
        return

    # useful constants
    # all of these colors are bolded
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[1;91m'
    YELLOW = '\033[1;93m'
    GREEN = '\033[1;92m'
    BLUE = '\033[1;96m'
    CYAN = '\033[1;94m'
    MAGENTA = '\033[1;95m'

    # only use colors if we're outputting to a terminal
    USE_COLORS = _sys.stdout.isatty()
    if not USE_COLORS:
        RESET = BOLD = RED = YELLOW = GREEN = BLUE = CYAN = MAGENTA = ''

    # generate the rainbow stars
    rainbow = [RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA]
    size = 78 // len(rainbow)
    stars = ''.join([color + '*' * size for color in rainbow])
    stars += RESET

    # do the actual output
    print(
        '\n'.join(
            [
                '',
                stars,
                BOLD,
                'Announcements go here.',
                RESET,
                # don't bold the suppression command
                'To suppress this message (and future announcements), run\n`touch {}`'.format(
                    noannounce_file
                ),
                stars,
            ]
        )
    )


def get_model_name(opt):
    """
    Get the model name from either `--model` or `--model-file`.
    """
    model = opt.get('model', None)
    if model is None:
        # try to get model name from model opt file
        model_file = opt.get('model_file', None)
        if model_file is not None:
            model_file = modelzoo_path(opt.get('datapath'), model_file)
            optfile = model_file + '.opt'
            if os.path.isfile(optfile):
                new_opt = Opt.load(optfile)
                model = new_opt.get('model', None)
    return model


def str2none(value: str):
    """
    If the value is a variant of `none`, return None.

    Otherwise, return the original value.
    """
    if value.lower() == 'none':
        return None
    else:
        return value


def str2bool(value):
    """
    Convert 'yes', 'false', '1', etc.

    into a boolean.
    """
    v = value.lower()
    if v in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2floats(s):
    """
    Look for single float or comma-separated floats.
    """
    return tuple(float(f) for f in s.split(','))


def str2multitask_weights(s):
    if s == 'stochastic':
        return s
    else:
        return str2floats(s)


def str2class(value):
    """
    From import path string, returns the class specified.

    For example, the string 'parlai.agents.drqa.drqa:SimpleDictionaryAgent' returns
    <class 'parlai.agents.drqa.drqa.SimpleDictionaryAgent'>.
    """
    if ':' not in value:
        raise RuntimeError('Use a colon before the name of the class.')
    name = value.split(':')
    module = importlib.import_module(name[0])
    return getattr(module, name[1])


def class2str(value):
    """
    Inverse of params.str2class().
    """
    s = str(value)
    s = s[s.find('\'') + 1 : s.rfind('\'')]  # pull out import path
    s = ':'.join(s.rsplit('.', 1))  # replace last period with ':'
    return s


def fix_underscores(args):
    """
    Convert underscores to hyphens in args.

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


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    Produce a custom-formatted `--help` option.

    See https://goo.gl/DKtHb5 for details.
    """

    def __init__(self, *args, **kwargs):
        kwargs['max_help_position'] = 6
        kwargs['width'] = 80
        super().__init__(*args, **kwargs)

    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ', '.join(action.option_strings) + ' ' + args_string

    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (default: %(default)s)'
        if (
            hasattr(action, 'recommended')
            and action.recommended
            and action.recommended != action.default
        ):
            help += '(recommended: %(recommended)s)'
            help = help.replace(')(recommended', ', recommended')
        return help


class ParlaiParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI argument parser.

    Pseudo-extension of ``argparse`` which sets a number of parameters
    for the ParlAI framework. More options can be added specific to other
    modules by passing this object and calling ``add_arg()`` or
    ``add_argument()`` on it.

    For example, see ``parlai.core.dict.DictionaryAgent.add_cmdline_args``.

    :param add_parlai_args:
        (default True) initializes the default arguments for ParlAI
        package, including the data download paths and task arguments.
    :param add_model_args:
        (default False) initializes the default arguments for loading
        models, including initializing arguments from that model.
    """

    def __init__(
        self, add_parlai_args=True, add_model_args=False, description='ParlAI parser'
    ):
        """
        Initialize the ParlAI argparser.
        """
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=CustomHelpFormatter,
            add_help=add_parlai_args,
        )
        self.register('type', 'nonestr', str2none)
        self.register('type', 'bool', str2bool)
        self.register('type', 'floats', str2floats)
        self.register('type', 'multitask_weights', str2multitask_weights)
        self.register('type', 'class', str2class)
        self.parlai_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ['PARLAI_HOME'] = self.parlai_home

        self.add_arg = self.add_argument

        # remember which args were specified on the command line
        self.overridable = {}

        if add_parlai_args:
            self.add_parlai_args()
        if add_model_args:
            self.add_model_args()

    def add_parlai_data_path(self, argument_group=None):
        """
        Add --datapath CLI arg.
        """
        if argument_group is None:
            argument_group = self
        argument_group.add_argument(
            '-dp',
            '--datapath',
            default=None,
            help='path to datasets, defaults to {parlai_dir}/data',
        )

    def add_mturk_args(self):
        """
        Add standard mechanical turk arguments.
        """
        mturk = self.add_argument_group('Mechanical Turk')
        default_log_path = os.path.join(self.parlai_home, 'logs', 'mturk')
        mturk.add_argument(
            '--mturk-log-path',
            default=default_log_path,
            help='path to MTurk logs, defaults to {parlai_dir}/logs/mturk',
        )
        mturk.add_argument(
            '-t',
            '--task',
            help='MTurk task, e.g. "qa_data_collection" or "model_evaluator"',
        )
        mturk.add_argument(
            '-nc',
            '--num-conversations',
            default=1,
            type=int,
            help='number of conversations you want to create for this task',
        )
        mturk.add_argument(
            '--unique',
            dest='unique_worker',
            default=False,
            action='store_true',
            help='enforce that no worker can work on your task twice',
        )
        mturk.add_argument(
            '--max-hits-per-worker',
            dest='max_hits_per_worker',
            default=0,
            type=int,
            help='Max number of hits each worker can perform during current group run',
        )
        mturk.add_argument(
            '--unique-qual-name',
            dest='unique_qual_name',
            default=None,
            type=str,
            help='qualification name to use for uniqueness between HITs',
        )
        mturk.add_argument(
            '-r',
            '--reward',
            default=0.05,
            type=float,
            help='reward for each worker for finishing the conversation, '
            'in US dollars',
        )
        mturk.add_argument(
            '--sandbox',
            dest='is_sandbox',
            action='store_true',
            help='submit the HITs to MTurk sandbox site',
        )
        mturk.add_argument(
            '--live',
            dest='is_sandbox',
            action='store_false',
            help='submit the HITs to MTurk live site',
        )
        mturk.add_argument(
            '--debug',
            dest='is_debug',
            action='store_true',
            help='print and log all server interactions and messages',
        )
        mturk.add_argument(
            '--verbose',
            dest='verbose',
            action='store_true',
            help='print all messages sent to and from Turkers',
        )
        mturk.add_argument(
            '--hard-block',
            dest='hard_block',
            action='store_true',
            default=False,
            help='Hard block disconnecting Turkers from all of your HITs',
        )
        mturk.add_argument(
            '--log-level',
            dest='log_level',
            type=int,
            default=20,
            help='importance level for what to put into the logs. the lower '
            'the level the more that gets logged. values are 0-50',
        )
        mturk.add_argument(
            '--disconnect-qualification',
            dest='disconnect_qualification',
            default=None,
            help='Qualification to use for soft blocking users for '
            'disconnects. By default '
            'turkers are never blocked, though setting this will allow '
            'you to filter out turkers that have disconnected too many '
            'times on previous HITs where this qualification was set.',
        )
        mturk.add_argument(
            '--block-qualification',
            dest='block_qualification',
            default=None,
            help='Qualification to use for soft blocking users. This '
            'qualification is granted whenever soft_block_worker is '
            'called, and can thus be used to filter workers out from a '
            'single task or group of tasks by noted performance.',
        )
        mturk.add_argument(
            '--count-complete',
            dest='count_complete',
            default=False,
            action='store_true',
            help='continue until the requested number of conversations are '
            'completed rather than attempted',
        )
        mturk.add_argument(
            '--allowed-conversations',
            dest='allowed_conversations',
            default=0,
            type=int,
            help='number of concurrent conversations that one mturk worker '
            'is able to be involved in, 0 is unlimited',
        )
        mturk.add_argument(
            '--max-connections',
            dest='max_connections',
            default=30,
            type=int,
            help='number of HITs that can be launched at the same time, 0 is '
            'unlimited.',
        )
        mturk.add_argument(
            '--min-messages',
            dest='min_messages',
            default=0,
            type=int,
            help='number of messages required to be sent by MTurk agent when '
            'considering whether to approve a HIT in the event of a '
            'partner disconnect. I.e. if the number of messages '
            'exceeds this number, the turker can submit the HIT.',
        )
        mturk.add_argument(
            '--local',
            dest='local',
            default=False,
            action='store_true',
            help='Run the server locally on this server rather than setting up'
            ' a heroku server.',
        )
        mturk.add_argument(
            '--hobby',
            dest='hobby',
            default=False,
            action='store_true',
            help='Run the heroku server on the hobby tier.',
        )
        mturk.add_argument(
            '--max-time',
            dest='max_time',
            default=0,
            type=int,
            help='Maximum number of seconds per day that a worker is allowed '
            'to work on this assignment',
        )
        mturk.add_argument(
            '--max-time-qual',
            dest='max_time_qual',
            default=None,
            help='Qualification to use to share the maximum time requirement '
            'with other runs from other machines.',
        )
        mturk.add_argument(
            '--heroku-team',
            dest='heroku_team',
            default=None,
            help='Specify Heroku team name to use for launching Dynos.',
        )
        mturk.add_argument(
            '--tmp-dir',
            dest='tmp_dir',
            default=None,
            help='Specify location to use for scratch builds and such.',
        )

        # it helps to indicate to agents that they're in interactive mode, and
        # can avoid some otherwise troublesome behavior (not having candidates,
        # sharing self.replies, etc).
        mturk.set_defaults(interactive_mode=True)

        mturk.set_defaults(is_sandbox=True)
        mturk.set_defaults(is_debug=False)
        mturk.set_defaults(verbose=False)

    def add_chatservice_args(self):
        """
        Arguments for all chat services.
        """
        args = self.add_argument_group('Chat Services')
        args.add_argument(
            '--debug',
            dest='is_debug',
            action='store_true',
            help='print and log all server interactions and messages',
        )
        args.add_argument(
            '--config-path',
            default=None,
            type=str,
            help='/path/to/config/file for a given task.',
        )
        args.add_argument(
            '--password',
            dest='password',
            type=str,
            default=None,
            help='Require a password for entry to the bot',
        )

    def add_websockets_args(self):
        """
        Add websocket arguments.
        """
        self.add_chatservice_args()
        websockets = self.add_argument_group('Websockets')
        websockets.add_argument(
            '--port', default=35496, type=int, help='Port to run the websocket handler'
        )

    def add_messenger_args(self):
        """
        Add Facebook Messenger arguments.
        """
        self.add_chatservice_args()
        messenger = self.add_argument_group('Facebook Messenger')
        messenger.add_argument(
            '--verbose',
            dest='verbose',
            action='store_true',
            help='print all messages sent to and from Turkers',
        )
        messenger.add_argument(
            '--log-level',
            dest='log_level',
            type=int,
            default=20,
            help='importance level for what to put into the logs. the lower '
            'the level the more that gets logged. values are 0-50',
        )
        messenger.add_argument(
            '--force-page-token',
            dest='force_page_token',
            action='store_true',
            help='override the page token stored in the cache for a new one',
        )
        messenger.add_argument(
            '--bypass-server-setup',
            dest='bypass_server_setup',
            action='store_true',
            default=False,
            help='should bypass traditional server and socket setup',
        )
        messenger.add_argument(
            '--local',
            dest='local',
            action='store_true',
            default=False,
            help='Run the server locally on this server rather than setting up'
            ' a heroku server.',
        )

        messenger.set_defaults(is_debug=False)
        messenger.set_defaults(verbose=False)

    def add_parlai_args(self, args=None):
        """
        Add common ParlAI args across all scripts.
        """
        parlai = self.add_argument_group('Main ParlAI Arguments')
        parlai.add_argument(
            '-o',
            '--init-opt',
            default=None,
            help='Path to json file of options. '
            'Note: Further Command-line arguments override file-based options.',
        )
        parlai.add_argument(
            '-v',
            '--show-advanced-args',
            action='store_true',
            help='Show hidden command line options (advanced users only)',
        )
        parlai.add_argument(
            '-t', '--task', help='ParlAI task(s), e.g. "babi:Task1" or "babi,cbt"'
        )
        parlai.add_argument(
            '--download-path',
            default=None,
            hidden=True,
            help='path for non-data dependencies to store any needed files.'
            'defaults to {parlai_dir}/downloads',
        )
        parlai.add_argument(
            '--loglevel',
            default='info',
            hidden=True,
            choices=logging.get_all_levels(),
            help='Logging level',
        )
        parlai.add_argument(
            '-dt',
            '--datatype',
            default='train',
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
                'test:stream',
            ],
            help='choose from: train, train:ordered, valid, test. to stream '
            'data add ":stream" to any option (e.g., train:stream). '
            'by default: train is random with replacement, '
            'valid is ordered, test is ordered.',
        )
        parlai.add_argument(
            '-im',
            '--image-mode',
            default='raw',
            type=str,
            help='image preprocessor to use. default is "raw". set to "none" '
            'to skip image loading.',
            hidden=True,
        )
        parlai.add_argument(
            '-nt',
            '--numthreads',
            default=1,
            type=int,
            help='number of threads. Used for hogwild if batchsize is 1, else '
            'for number of threads in threadpool loading,',
        )
        parlai.add_argument(
            '--hide-labels',
            default=False,
            type='bool',
            hidden=True,
            help='default (False) moves labels in valid and test sets to the '
            'eval_labels field. If True, they are hidden completely.',
        )
        parlai.add_argument(
            '-mtw',
            '--multitask-weights',
            type='multitask_weights',
            default=[1],
            help=(
                'list of floats, one for each task, specifying '
                'the probability of drawing the task in multitask case. You may also '
                'provide "stochastic" to simulate simple concatenation.'
            ),
            hidden=True,
        )
        parlai.add_argument(
            '-bs',
            '--batchsize',
            default=1,
            type=int,
            help='batch size for minibatch training schemes',
        )
        parlai.add_argument(
            '-dynb',
            '--dynamic-batching',
            default=None,
            type='nonestr',
            choices={None, 'full', 'batchsort'},
            help='Use dynamic batching',
        )
        self.add_parlai_data_path(parlai)

    def add_distributed_training_args(self):
        """
        Add CLI args for distributed training.
        """
        grp = self.add_argument_group('Distributed Training')
        grp.add_argument(
            '--distributed-world-size', type=int, help='Number of workers.'
        )
        grp.add_argument(
            '--verbose',
            type='bool',
            default=False,
            help='All workers print output.',
            hidden=True,
        )
        return grp

    def add_model_args(self):
        """
        Add arguments related to models such as model files.
        """
        model_args = self.add_argument_group('ParlAI Model Arguments')
        model_args.add_argument(
            '-m',
            '--model',
            default=None,
            help='the model class name. can match parlai/agents/<model> for '
            'agents in that directory, or can provide a fully specified '
            'module for `from X import Y` via `-m X:Y` '
            '(e.g. `-m parlai.agents.seq2seq.seq2seq:Seq2SeqAgent`)',
        )
        model_args.add_argument(
            '-mf',
            '--model-file',
            default=None,
            help='model file name for loading and saving models',
        )
        model_args.add_argument(
            '-im',
            '--init-model',
            default=None,
            type=str,
            help='load model weights and dict from this file',
        )
        model_args.add_argument(
            '--dict-class', hidden=True, help='the class of the dictionary agent uses'
        )

    def add_model_subargs(self, model):
        """
        Add arguments specific to a particular model.
        """
        agent = load_agent_module(model)
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
        """
        Add arguments specific to the specified task.
        """
        for t in ids_to_tasks(task).split(','):
            agent = load_teacher_module(t)
            try:
                if hasattr(agent, 'add_cmdline_args'):
                    agent.add_cmdline_args(self)
            except argparse.ArgumentError:
                # already added
                pass

    def add_world_args(self, task, interactive_task, selfchat_task):
        """
        Add arguments specific to the world.
        """
        world_class = load_world_module(
            task, interactive_task=interactive_task, selfchat_task=selfchat_task
        )
        if world_class is not None and hasattr(world_class, 'add_cmdline_args'):
            try:
                world_class.add_cmdline_args(self)
            except argparse.ArgumentError:
                # already added
                pass

    def add_image_args(self, image_mode):
        """
        Add additional arguments for handling images.
        """
        try:
            parlai = self.add_argument_group('ParlAI Image Preprocessing Arguments')
            parlai.add_argument(
                '--image-size',
                type=int,
                default=256,
                help='resizing dimension for images',
                hidden=True,
            )
            parlai.add_argument(
                '--image-cropsize',
                type=int,
                default=224,
                help='crop dimension for images',
                hidden=True,
            )
        except argparse.ArgumentError:
            # already added
            pass

    def add_extra_args(self, args=None):
        """
        Add more args depending on how known args are set.
        """
        parsed = vars(self.parse_known_args(args, nohelp=True)[0])
        # Also load extra args options if a file is given.
        if parsed.get('init_opt', None) is not None:
            self._load_known_opts(parsed.get('init_opt'), parsed)
        parsed = self._infer_datapath(parsed)

        # find which image mode specified if any, and add additional arguments
        image_mode = parsed.get('image_mode', None)
        if image_mode is not None and image_mode != 'no_image_model':
            self.add_image_args(image_mode)

        # find which task specified if any, and add its specific arguments
        task = parsed.get('task', None)
        if task is not None:
            self.add_task_args(task)
        evaltask = parsed.get('evaltask', None)
        if evaltask is not None:
            self.add_task_args(evaltask)

        # find which model specified if any, and add its specific arguments
        model = get_model_name(parsed)
        if model is not None:
            self.add_model_subargs(model)

        # add world args, if we know a priori which world is being used
        if task is not None:
            self.add_world_args(
                task,
                parsed.get('interactive_task', False),
                parsed.get('selfchat_task', False),
            )

        # reset parser-level defaults over any model-level defaults
        try:
            self.set_defaults(**self._defaults)
        except AttributeError:
            raise RuntimeError(
                'Please file an issue on github that argparse '
                'got an attribute error when parsing.'
            )

    def parse_known_args(self, args=None, namespace=None, nohelp=False):
        """
        Parse known args to ignore help flag.
        """
        if args is None:
            # args default to the system args
            args = _sys.argv[1:]
        args = fix_underscores(args)

        if nohelp:
            # ignore help
            args = [a for a in args if a != '-h' and a != '--help']
        return super().parse_known_args(args, namespace)

    def _load_known_opts(self, optfile, parsed):
        """
        Pull in CLI args for proper models/tasks/etc.

        Called before args are parsed; ``_load_opts`` is used for actually overriding
        opts after they are parsed.
        """
        new_opt = Opt.load(optfile)
        for key, value in new_opt.items():
            # existing command line parameters take priority.
            if key not in parsed or parsed[key] is None:
                parsed[key] = value

    def _load_opts(self, opt):
        optfile = opt.get('init_opt')
        new_opt = Opt.load(optfile)
        for key, value in new_opt.items():
            # existing command line parameters take priority.
            if key not in opt:
                raise RuntimeError(
                    'Trying to set opt from file that does not exist: ' + str(key)
                )
            if key not in opt['override']:
                opt[key] = value
                opt['override'][key] = value

    def _infer_datapath(self, opt):
        """
        Set the value for opt['datapath'] and opt['download_path'].

        Sets the value for opt['datapath'] and opt['download_path'], correctly
        respecting environmental variables and the default.
        """
        # set environment variables
        # Priority for setting the datapath (same applies for download_path):
        # --datapath -> os.environ['PARLAI_DATAPATH'] -> <self.parlai_home>/data
        if opt.get('download_path'):
            os.environ['PARLAI_DOWNPATH'] = opt['download_path']
        elif os.environ.get('PARLAI_DOWNPATH') is None:
            os.environ['PARLAI_DOWNPATH'] = os.path.join(self.parlai_home, 'downloads')
        if opt.get('datapath'):
            os.environ['PARLAI_DATAPATH'] = opt['datapath']
        elif os.environ.get('PARLAI_DATAPATH') is None:
            os.environ['PARLAI_DATAPATH'] = os.path.join(self.parlai_home, 'data')

        opt['download_path'] = os.environ['PARLAI_DOWNPATH']
        opt['datapath'] = os.environ['PARLAI_DATAPATH']

        return opt

    def _process_args_to_opts(self, args_that_override: Optional[List[str]] = None):
        self.opt = Opt(vars(self.args))

        # custom post-parsing
        self.opt['parlai_home'] = self.parlai_home
        self.opt = self._infer_datapath(self.opt)

        # set all arguments specified in command line as overridable
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

        if args_that_override is None:
            args_that_override = _sys.argv[1:]

        for i in range(len(args_that_override)):
            if args_that_override[i] in option_strings_dict:
                if args_that_override[i] in store_true:
                    self.overridable[option_strings_dict[args_that_override[i]]] = True
                elif args_that_override[i] in store_false:
                    self.overridable[option_strings_dict[args_that_override[i]]] = False
                elif (
                    i < len(args_that_override) - 1
                    and args_that_override[i + 1][:1] != '-'
                ):
                    key = option_strings_dict[args_that_override[i]]
                    self.overridable[key] = self.opt[key]
        self.opt['override'] = self.overridable

        # load opts if a file is provided.
        if self.opt.get('init_opt', None) is not None:
            self._load_opts(self.opt)

        # map filenames that start with 'zoo:' to point to the model zoo dir
        options_to_change = {'model_file', 'dict_file', 'bpe_vocab', 'bpe_merge'}
        for each_key in options_to_change:
            if self.opt.get(each_key) is not None:
                self.opt[each_key] = modelzoo_path(
                    self.opt.get('datapath'), self.opt[each_key]
                )
            if self.opt['override'].get(each_key) is not None:
                # also check override
                self.opt['override'][each_key] = modelzoo_path(
                    self.opt.get('datapath'), self.opt['override'][each_key]
                )

        # add start time of an experiment
        self.opt['starttime'] = datetime.datetime.today().strftime('%b%d_%H-%M')

    def parse_and_process_known_args(self, args=None):
        """
        Parse provided arguments and return parlai opts and unknown arg list.

        Runs the same arg->opt parsing that parse_args does, but doesn't throw an error
        if the args being parsed include additional command line arguments that parlai
        doesn't know what to do with.
        """
        self.args, unknowns = super().parse_known_args(args=args)
        self._process_args_to_opts(args)
        return self.opt, unknowns

    def parse_args(self, args=None, namespace=None, print_args=True):
        """
        Parse the provided arguments and returns a dictionary of the ``args``.

        We specifically remove items with ``None`` as values in order to support the
        style ``opt.get(key, default)``, which would otherwise return ``None``.
        """
        self.add_extra_args(args)
        self.args = super().parse_args(args=args)

        self._process_args_to_opts(args)

        if print_args:
            self.print_args()
            if GIT_AVAILABLE:
                print_git_commit()
            print_announcements(self.opt)

        logging.set_log_level(self.opt.get('loglevel', 'info').upper())

        return self.opt

    def _kwargs_to_str_args(self, **kwargs):
        """
        Attempt to map from python-code kwargs into CLI args.

        e.g. model_file -> --model-file.

        Works with short options too, like t="convai2".
        """

        # we have to do this large block of repetitive code twice, the first
        # round is basically just to become aware of anything that would have
        # been added by add_extra_args
        kwname_to_action = {}
        for action in self._actions:
            if action.dest == 'help':
                # no help allowed
                continue
            for option_string in action.option_strings:
                kwname = option_string.lstrip('-').replace('-', '_')
                assert (kwname not in kwname_to_action) or (
                    kwname_to_action[kwname] is action
                ), f"No duplicate names! ({kwname}, {kwname_to_action[kwname]}, {action})"
                kwname_to_action[kwname] = action

        string_args = []
        for kwname, value in kwargs.items():
            if kwname not in kwname_to_action:
                # best guess, we need to delay it. hopefully this gets added
                # during add_kw_Args
                continue
            action = kwname_to_action[kwname]
            last_option_string = action.option_strings[-1]
            if isinstance(action, argparse._StoreTrueAction) and bool(value):
                string_args.append(last_option_string)
            elif isinstance(action, argparse._StoreAction) and action.nargs is None:
                string_args.append(last_option_string)
                string_args.append(str(value))
            elif isinstance(action, argparse._StoreAction) and action.nargs in '*+':
                string_args.append(last_option_string)
                string_args.extend([str(v) for v in value])
            else:
                raise TypeError(f"Don't know what to do with {action}")

        # become aware of any extra args that might be specified if the user
        # provides something like model="transformer/generator".
        self.add_extra_args(string_args)

        # do it again, this time knowing about ALL args.
        kwname_to_action = {}
        for action in self._actions:
            if action.dest == 'help':
                # no help allowed
                continue
            for option_string in action.option_strings:
                kwname = option_string.lstrip('-').replace('-', '_')
                assert (kwname not in kwname_to_action) or (
                    kwname_to_action[kwname] is action
                ), f"No duplicate names! ({kwname}, {kwname_to_action[kwname]}, {action})"
                kwname_to_action[kwname] = action

        string_args = []
        for kwname, value in kwargs.items():
            # note we don't have the if kwname not in kwname_to_action here.
            # it MUST appear, or else we legitimately should be throwing a KeyError
            # because user has provided an unspecified option
            action = kwname_to_action[kwname]
            last_option_string = action.option_strings[-1]
            if isinstance(action, argparse._StoreTrueAction) and bool(value):
                string_args.append(last_option_string)
            elif isinstance(action, argparse._StoreAction) and action.nargs is None:
                string_args.append(last_option_string)
                string_args.append(str(value))
            elif isinstance(action, argparse._StoreAction) and action.nargs in '*+':
                string_args.append(last_option_string)
                string_args.extend([str(v) for v in value])
            else:
                raise TypeError(f"Don't know what to do with {action}")

        return string_args

    def parse_kwargs(self, **kwargs):
        """
        Parse kwargs, with type checking etc.
        """
        # hack: capture any error messages without raising a SystemExit
        def _captured_error(msg):
            raise ValueError(msg)

        old_error = self.error
        self.error = _captured_error
        try:
            string_args = self._kwargs_to_str_args(**kwargs)
            return self.parse_args(args=string_args, print_args=False)
        finally:
            self.error = old_error

    def print_args(self):
        """
        Print out all the arguments in this parser.
        """
        if not self.opt:
            self.parse_args(print_args=False)
        values = {}
        for key, value in self.opt.items():
            values[str(key)] = str(value)
        for group in self._action_groups:
            group_dict = {
                a.dest: getattr(self.args, a.dest, None) for a in group._group_actions
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
        """
        Set overridable kwargs.
        """
        self.set_defaults(**kwargs)
        for k, v in kwargs.items():
            self.overridable[k] = v

    @property
    def show_advanced_args(self):
        """
        Check if we should show arguments marked as hidden.
        """
        if hasattr(self, '_show_advanced_args'):
            return self._show_advanced_args
        known_args, _ = self.parse_known_args(nohelp=True)
        if hasattr(known_args, 'show_advanced_args'):
            self._show_advanced_args = known_args.show_advanced_args
        else:
            self._show_advanced_args = True
        return self._show_advanced_args

    def _handle_custom_options(self, kwargs):
        """
        Handle custom parlai options.

        Includes hidden, recommended. Future may include no_save and no_override.
        """
        action_attr = {}
        if 'recommended' in kwargs:
            rec = kwargs.pop('recommended')
            action_attr['recommended'] = rec
        action_attr['hidden'] = kwargs.get('hidden', False)
        if 'hidden' in kwargs:
            hidden = kwargs.pop('hidden')
            if hidden:
                kwargs['help'] = argparse.SUPPRESS
        if 'type' in kwargs and kwargs['type'] is bool:
            # common error, we really want simple form
            kwargs['type'] = 'bool'
        return kwargs, action_attr

    def add_argument(self, *args, **kwargs):
        """
        Override to convert underscores to hyphens for consistency.
        """
        kwargs, newattr = self._handle_custom_options(kwargs)
        action = super().add_argument(*fix_underscores(args), **kwargs)
        for k, v in newattr.items():
            setattr(action, k, v)
        return action

    def add_argument_group(self, *args, **kwargs):
        """
        Override to make arg groups also convert underscores to hyphens.
        """
        arg_group = super().add_argument_group(*args, **kwargs)
        original_add_arg = arg_group.add_argument

        def ag_add_argument(*args, **kwargs):
            kwargs, newattr = self._handle_custom_options(kwargs)
            action = original_add_arg(*fix_underscores(args), **kwargs)
            for k, v in newattr.items():
                setattr(action, k, v)
            return action

        arg_group.add_argument = ag_add_argument  # override _ => -
        arg_group.add_argument_group = self.add_argument_group
        return arg_group

    def error(self, message):
        """
        Override to print custom error message.
        """
        self.print_help()
        _sys.stderr.write('\nParse Error: %s\n' % message)
        _sys.exit(2)
