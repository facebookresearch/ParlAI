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
from parlai.utils.io import PathManager

from typing import List, Optional


def print_git_commit():
    """
    Print the current git commit of ParlAI and parlai_internal.
    """
    if not GIT_AVAILABLE:
        return
    root = os.path.dirname(os.path.dirname(parlai.__file__))
    internal_root = os.path.join(root, 'parlai_internal')
    fb_root = os.path.join(root, 'parlai_fb')
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

    try:
        git_ = git.Git(fb_root)
        fb_commit = git_.rev_parse('HEAD')
        logging.info(f'Current fb commit: {fb_commit}')
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
    if PathManager.exists(noannounce_file):
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
            if PathManager.exists(optfile):
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

    For example, the string
    'parlai.agents.hugging_face.dict:Gpt2DictionaryAgent' returns
    <class 'parlai.agents.hugging_face.dict.Gpt2DictionaryAgent'>.
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


class _HelpAllAction(argparse._HelpAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if hasattr(parser, '_unsuppress_hidden'):
            parser._unsuppress_hidden()
        super().__call__(parser, namespace, values, option_string=option_string)


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    Produce a custom-formatted `--help` option.

    See https://goo.gl/DKtHb5 for details.
    """

    def __init__(self, *args, **kwargs):
        if 'max_help_position' not in kwargs:
            kwargs['max_help_position'] = 8
        super().__init__(*args, **kwargs)

    def _fill_text(self, text, width, indent):
        # used to ensure that argparse doesn't word-wrap our descriptions of
        # commands. mostly useful for the logo in the supercommand.
        return ''.join(indent + line for line in text.splitlines(keepends=True))

    def _iter_indented_subactions(self, action):
        # used in superscript parser to hide "hidden" commands.
        retval = super()._iter_indented_subactions(action)
        if isinstance(action, argparse._SubParsersAction):
            retval = [x for x in retval if x.help != argparse.SUPPRESS]
        return retval

    def _format_action_invocation(self, action):
        # used to suppress one utterance in the super parser.
        if isinstance(action, argparse._SubParsersAction):
            return ""
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ', '.join(action.option_strings) + ' ' + args_string

    def _get_help_string(self, action):
        """
        Help string that (almost) always inserts %(default)s.
        """
        help = action.help
        if (
            '%(default)' in action.help
            or not isinstance(action, argparse._StoreAction)
            or action.default is argparse.SUPPRESS
        ):
            return help

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

    For an example, see ``parlai.core.dict.DictionaryAgent.add_cmdline_args``.

    :param add_parlai_args:
        (default True) initializes the default arguments for ParlAI
        package, including the data download paths and task arguments.
    :param add_model_args:
        (default False) initializes the default arguments for loading
        models, including initializing arguments from that model.
    """

    def __init__(
        self, add_parlai_args=True, add_model_args=False, description=None, **kwargs
    ):
        """
        Initialize the ParlAI parser.
        """
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = CustomHelpFormatter

        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            add_help=True,
            **kwargs,
        )
        self.register('action', 'helpall', _HelpAllAction)
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
        self.add_argument(
            '--helpall',
            action='helpall',
            help='Show usage, including advanced arguments.',
        )
        parlai = self.add_argument_group('Main ParlAI Arguments')
        parlai.add_argument(
            '-o',
            '--init-opt',
            default=None,
            help='Path to json file of options. '
            'Note: Further Command-line arguments override file-based options.',
        )
        parlai.add_argument(
            '--allow-missing-init-opts',
            type='bool',
            default=False,
            help=(
                'Warn instead of raising if an argument passed in with --init-opt is '
                'not in the target opt.'
            ),
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
            metavar='DATATYPE',
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
            'by default train is random with replacement, '
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
        parlai.add_argument(
            '-v',
            '--verbose',
            dest='verbose',
            action='store_true',
            help='Print all messages',
        )
        parlai.add_argument(
            '--debug',
            dest='is_debug',
            action='store_true',
            help='Enables some debug behavior',
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
            '--ddp-backend',
            # TODO: add in zero3. https://github.com/facebookresearch/ParlAI/issues/3753
            choices=['ddp', 'zero2'],
            default='ddp',
            help=(
                'Distributed backend. Zero2 can be faster but is more experimental. '
                'DDP is the most tested.'
            ),
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
            help='Initialize model weights and dict from this file',
        )
        model_args.add_argument(
            '--dict-class', hidden=True, help='the class of the dictionary agent uses'
        )

    def add_model_subargs(self, model: str, partial: Opt):
        """
        Add arguments specific to a particular model.
        """
        agent = load_agent_module(model)
        try:
            if hasattr(agent, 'add_cmdline_args'):
                agent.add_cmdline_args(self, partial)
        except TypeError as typ:
            raise TypeError(
                f"Agent '{model}' appears to have signature "
                "add_cmdline_args(argparser) but we have updated the signature "
                "to add_cmdline_args(argparser, partial_opt). For details, see "
                "https://github.com/facebookresearch/ParlAI/pull/3328."
            ) from typ
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

    def add_task_args(self, task: str, partial: Opt):
        """
        Add arguments specific to the specified task.
        """
        for t in ids_to_tasks(task).split(','):
            agent = load_teacher_module(t)
            try:
                if hasattr(agent, 'add_cmdline_args'):
                    agent.add_cmdline_args(self, partial)
            except TypeError as typ:
                raise TypeError(
                    f"Task '{task}' appears to have signature "
                    "add_cmdline_args(argparser) but we have updated the signature "
                    "to add_cmdline_args(argparser, partial_opt). For details, see "
                    "https://github.com/facebookresearch/ParlAI/pull/3328."
                ) from typ
            except argparse.ArgumentError:
                # already added
                pass

    def add_world_args(
        self,
        task: str,
        interactive_task: Optional[str],
        selfchat_task: Optional[str],
        partial: Opt,
    ):
        """
        Add arguments specific to the world.
        """
        world_class = load_world_module(
            task, interactive_task=interactive_task, selfchat_task=selfchat_task
        )
        if world_class is not None and hasattr(world_class, 'add_cmdline_args'):
            try:
                world_class.add_cmdline_args(self, partial)
            except argparse.ArgumentError:
                # already added
                pass
            except TypeError:
                raise TypeError(
                    f"World '{task}' appears to have signature "
                    "add_cmdline_args(argparser) but we have updated the signature "
                    "to add_cmdline_args(argparser, partial_opt). For details, see "
                    "https://github.com/facebookresearch/ParlAI/pull/3328."
                )

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
        if parsed.get('init_opt') is not None:
            try:
                self._load_known_opts(parsed.get('init_opt'), parsed)
            except FileNotFoundError:
                # don't die if -o isn't found here. See comment in second call
                # later on.
                pass
        parsed = self._infer_datapath(parsed)

        partial = Opt(parsed)

        # find which image mode specified if any, and add additional arguments
        image_mode = parsed.get('image_mode', None)
        if image_mode is not None and image_mode != 'no_image_model':
            self.add_image_args(image_mode)

        # find which task specified if any, and add its specific arguments
        task = parsed.get('task', None)
        if task is not None:
            self.add_task_args(task, partial)
        evaltask = parsed.get('evaltask', None)
        if evaltask is not None:
            self.add_task_args(evaltask, partial)

        # find which model specified if any, and add its specific arguments
        model = get_model_name(parsed)
        if model is not None:
            self.add_model_subargs(model, partial)

        # add world args, if we know a priori which world is being used
        if task is not None:
            self.add_world_args(
                task,
                parsed.get('interactive_task', False),
                parsed.get('selfchat_task', False),
                partial,
            )

        # reparse args now that we've inferred some things.  specifically helps
        # with a misparse of `-opt` as `-o pt`, which causes opt loading to
        # try to load the file "pt" which doesn't exist.
        # After adding model arguments, -opt becomes known (it's in TorchAgent),
        # and we parse the `-opt` value correctly.
        parsed = vars(self.parse_known_args(args, nohelp=True)[0])
        if parsed.get('init_opt') is not None:
            self._load_known_opts(parsed.get('init_opt'), parsed)

        # reset parser-level defaults over any model-level defaults
        try:
            self.set_defaults(**self._defaults)
        except AttributeError:
            raise RuntimeError(
                'Please file an issue on github that argparse '
                'got an attribute error when parsing.'
            )

    def _handle_single_dash_parsearg(self, args, actions):
        if _sys.version_info >= (3, 8, 0):
            newargs = []
            for arg in args:
                darg = f'-{arg}'
                if arg.startswith('-') and not arg.startswith('--') and darg in actions:
                    newargs.append(darg)
                else:
                    newargs.append(arg)
            return newargs
        else:
            return args

    def parse_known_args(self, args=None, namespace=None, nohelp=False):
        """
        Parse known args to ignore help flag.
        """
        if args is None:
            # args default to the system args
            args = _sys.argv[1:]

        args = fix_underscores(args)
        # handle the single dash stuff. See _handle_single_dash_addarg for info
        actions = set()
        for action in self._actions:
            actions.update(action.option_strings)
        args = self._handle_single_dash_parsearg(args, actions)
        if nohelp:
            # ignore help
            args = [
                a
                for a in args
                if a != '-h' and a != '--help' and a != '--helpall' and a != '--h'
            ]
        return super().parse_known_args(args, namespace)

    def _load_known_opts(self, optfile, parsed):
        """
        Pull in CLI args for proper models/tasks/etc.

        Called before args are parsed; ``_load_opts`` is used for actually overriding
        opts after they are parsed.
        """
        new_opt = Opt.load_init(optfile)
        for key, value in new_opt.items():
            # existing command line parameters take priority.
            if key not in parsed or parsed[key] is None:
                parsed[key] = value

    def _load_opts(self, opt):
        optfile = opt.get('init_opt')
        new_opt = Opt.load_init(optfile)
        for key, value in new_opt.items():
            # existing command line parameters take priority.
            if key not in opt:
                if opt.get('allow_missing_init_opts', False):
                    logging.warning(
                        f'The "{key}" key in {optfile} will not be loaded, because it '
                        f'does not exist in the target opt.'
                    )
                else:
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
        if opt.get('datapath'):
            os.environ['PARLAI_DATAPATH'] = opt['datapath']
        elif os.environ.get('PARLAI_DATAPATH') is None:
            DEFAULT_DATAPATH = None
            try:
                # internal fbcode-wide default
                import parlai_fb

                DEFAULT_DATAPATH = parlai_fb.DEFAULT_DATAPATH
            except ImportError:
                pass
            if not DEFAULT_DATAPATH:
                # TODO: switch to ~/.parlai/
                DEFAULT_DATAPATH = os.path.join(self.parlai_home, 'data')
            os.environ['PARLAI_DATAPATH'] = DEFAULT_DATAPATH

        opt['datapath'] = os.environ['PARLAI_DATAPATH']

        return opt

    def _process_args_to_opts(self, args_that_override: Optional[List[str]] = None):
        self.opt = Opt(vars(self.args))
        extra_ag = []

        if '_subparser' in self.opt:
            # if using the super command, we need to be aware of the subcommand's
            # arguments when identifying things manually set by the user
            self.overridable.update(self.opt['_subparser'].overridable)
            extra_ag = self.opt.pop('_subparser')._action_groups

        # custom post-parsing
        self.opt['parlai_home'] = self.parlai_home
        self.opt = self._infer_datapath(self.opt)

        # set all arguments specified in command line as overridable
        option_strings_dict = {}
        store_true = []
        store_false = []
        for group in self._action_groups + extra_ag:
            for a in group._group_actions:
                if hasattr(a, 'option_strings'):
                    for option in a.option_strings:
                        option_strings_dict[option] = a.dest
                        if isinstance(a, argparse._StoreTrueAction):
                            store_true.append(option)
                        elif isinstance(a, argparse._StoreFalseAction):
                            store_false.append(option)

        if args_that_override is None:
            args_that_override = _sys.argv[1:]

        args_that_override = self._handle_single_dash_parsearg(
            fix_underscores(args_that_override), option_strings_dict.keys()
        )

        for i in range(len(args_that_override)):
            if args_that_override[i] in option_strings_dict:
                if args_that_override[i] in store_true:
                    self.overridable[option_strings_dict[args_that_override[i]]] = True
                elif args_that_override[i] in store_false:
                    self.overridable[option_strings_dict[args_that_override[i]]] = False
                elif (
                    i < len(args_that_override) - 1
                    and args_that_override[i + 1] not in option_strings_dict
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

    def parse_args(self, args=None, namespace=None, **kwargs):
        """
        Parse the provided arguments and returns a dictionary of the ``args``.

        We specifically remove items with ``None`` as values in order to support the
        style ``opt.get(key, default)``, which would otherwise return ``None``.
        """
        if 'print_args' in kwargs:
            logging.error(
                "You gave the print_args flag to parser.parse_args, but this is "
                "no longer supported. Use opt.log() to print the arguments"
            )
            del kwargs['print_args']
        self.add_extra_args(args)
        self.args = super().parse_args(args=args)

        self._process_args_to_opts(args)
        print_announcements(self.opt)

        assert '_subparser' not in self.opt

        return self.opt

    def _value2argstr(self, value) -> str:
        """
        Reverse-parse an opt value into one interpretable by argparse.
        """
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        else:
            return str(value)

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

        # since we can have options that depend on options, repeat until convergence
        string_args = []
        unparsed_args = set(kwargs.keys())
        while unparsed_args:
            string_args = []
            for kwname, value in kwargs.items():
                if kwname not in kwname_to_action:
                    # best guess, we need to delay it. hopefully this gets added
                    # during add_kw_Args
                    continue
                action = kwname_to_action[kwname]
                last_option_string = action.option_strings[-1]
                if isinstance(action, argparse._StoreTrueAction):
                    if bool(value):
                        string_args.append(last_option_string)
                elif isinstance(action, argparse._StoreAction) and action.nargs is None:
                    string_args.append(last_option_string)
                    string_args.append(self._value2argstr(value))
                elif isinstance(action, argparse._StoreAction) and action.nargs in '*+':
                    string_args.append(last_option_string)
                    string_args.extend([self._value2argstr(v) for v in value])
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

            new_unparsed_args = set()
            string_args = []
            for kwname, value in kwargs.items():
                if kwname not in kwname_to_action:
                    new_unparsed_args.add(kwname)
                    continue

                action = kwname_to_action[kwname]
                last_option_string = action.option_strings[-1]
                if isinstance(action, argparse._StoreTrueAction):
                    if bool(value):
                        string_args.append(last_option_string)
                elif isinstance(action, argparse._StoreAction) and action.nargs is None:
                    string_args.append(last_option_string)
                    string_args.append(self._value2argstr(value))
                elif isinstance(action, argparse._StoreAction) and action.nargs in '*+':
                    string_args.append(last_option_string)
                    # Special case: Labels
                    string_args.extend([self._value2argstr(v) for v in value])
                else:
                    raise TypeError(f"Don't know what to do with {action}")

            if new_unparsed_args == unparsed_args:
                # if we have converged to a fixed point with no improvements, we
                # truly found some unreachable args
                raise KeyError(
                    f'Failed to parse one or more kwargs: {", ".join(new_unparsed_args)}'
                )
            else:
                # We've seen some improvements on the number of unparsed args,
                # iterate again
                unparsed_args = new_unparsed_args

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
            return self.parse_args(args=string_args)
        finally:
            self.error = old_error

    def set_params(self, **kwargs):
        """
        Set overridable kwargs.
        """
        self.set_defaults(**kwargs)
        for k, v in kwargs.items():
            self.overridable[k] = v

    def _unsuppress_hidden(self):
        for action in self._actions:
            if hasattr(action, 'real_help'):
                action.help = action.real_help

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
        action_attr['real_help'] = kwargs.get('help', None)
        if 'hidden' in kwargs:
            if kwargs.pop('hidden'):
                kwargs['help'] = argparse.SUPPRESS

        if 'type' in kwargs and kwargs['type'] is bool:
            # common error, we really want simple form
            kwargs['type'] = 'bool'
        return kwargs, action_attr

    def _handle_single_dash_addarg(self, args):
        """
        Fixup argparse for parlai-style short args.

        In python 3.8, argparsing was changed such that short arguments are not
        required to have spaces after them. This causes our many short args to
        be misinterpetted by the parser. For example `-emb` gets parsed as
        `-e mb`.

        Here we rewrite them into long args to get around the nonsense.
        """
        if _sys.version_info < (3, 8, 0):
            # older python works fine
            return args

        # need the long options specified first, or `dest` will get set to
        # the short name on accident!
        out_long = []
        out_short = []
        for arg in args:
            if arg.startswith('-') and not arg.startswith('--'):
                out_short.append(f'-{arg}')
            else:
                out_long.append(arg)
        # keep long args in front so they are used for the destination
        return out_long + out_short

    def add_argument(self, *args, **kwargs):
        """
        Override to convert underscores to hyphens for consistency.
        """
        kwargs, newattr = self._handle_custom_options(kwargs)
        args = self._handle_single_dash_addarg(fix_underscores(args))
        action = super().add_argument(*args, **kwargs)
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
            args = self._handle_single_dash_addarg(fix_underscores(args))
            action = original_add_arg(*args, **kwargs)
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


def default(val, default):
    """
    shorthand for explicit None check for optional arguments.
    """
    return val if val is not None else default
