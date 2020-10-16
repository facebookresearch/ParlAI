#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
The general ParlAI Script interface.

An abstract class to help standardize the call to ParlAI scripts, enabling them to be
completed easily.

Also contains helper classes for loading scripts, etc.
"""

import io
import argparse
from typing import List, Optional, Dict, Any
import parlai
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser, CustomHelpFormatter
from abc import abstractmethod
import importlib
import pkgutil
import parlai.scripts
import parlai.utils.logging as logging
from parlai.core.loader import register_script, SCRIPT_REGISTRY  # noqa: F401


def setup_script_registry():
    """
    Loads the scripts so that @register_script is hit for all.
    """
    for module in pkgutil.iter_modules(parlai.scripts.__path__, 'parlai.scripts.'):
        importlib.import_module(module.name)


class ParlaiScript(object):
    """
    A ParlAI script is a standardized form of access.
    """

    parser: ParlaiParser

    @classmethod
    @abstractmethod
    def setup_args(cls) -> ParlaiParser:
        """
        Create the parser with args.
        """
        # we want to later deprecate this for add_cmdline_args
        pass

    def __init__(self, opt: Opt):
        self.opt = opt

    @abstractmethod
    def run(self):
        """
        The main method.

        Must be implemented by the script writer.
        """
        raise NotImplementedError()

    @classmethod
    def _run_kwargs(cls, kwargs: Dict[str, Any]):
        """
        Construct and run the script using kwargs, pseudo-parsing them.
        """
        parser = cls.setup_args()
        opt = parser.parse_kwargs(**kwargs)
        return cls._run_from_parser_and_opt(opt, parser)

    @classmethod
    def _run_args(cls, args: Optional[List[str]] = None):
        """
        Construct and run the script using args, defaulting to getting from CLI.
        """
        parser = cls.setup_args()
        opt = parser.parse_args(args=args)
        return cls._run_from_parser_and_opt(opt, parser)

    @classmethod
    def _run_from_parser_and_opt(cls, opt: Opt, parser: ParlaiParser):
        script = cls(opt)
        script.parser = parser
        return script.run()

    @classmethod
    def main(cls, *args, **kwargs):
        """
        Run the program, possibly with some given args.

        You may provide command line args in the form of strings, or
        options. For example:

        >>> MyScript.main(['--task', 'convai2'])
        >>> MyScript.main(task='convai2')

        You may not combine both args and kwargs.
        """
        assert not (bool(args) and bool(kwargs))
        if args:
            return cls._run_args(args)
        elif kwargs:
            return cls._run_kwargs(kwargs)
        else:
            return cls._run_args(None)

    @classmethod
    def help(cls, **kwargs):
        f = io.StringIO()
        parser = cls.setup_args()
        parser.prog = cls.__name__
        parser.add_extra_args(parser._kwargs_to_str_args(**kwargs))
        parser.print_help(f)
        return f.getvalue()


class _SupercommandParser(ParlaiParser):
    """
    Specialty ParlAI parser used for the supercommand.

    Contains some special behavior.
    """

    def __init__(self, *args, **kwargs):
        # used to target help messages more correctly, see GH #3182
        self._help_subparser = None

        from parlai.utils.strings import colorize

        logo = ""
        logo += colorize('       _', 'red') + "\n"
        logo += colorize('      /', 'red') + colorize('"', 'brightblack')
        logo += colorize(")", "yellow") + "\n"
        logo += colorize('     //', 'red') + colorize(')', 'yellow') + '\n'
        logo += colorize('  ==', 'green')
        logo += colorize("/", 'blue') + colorize('/', 'red') + colorize("'", 'yellow')
        logo += colorize("===", 'green') + " ParlAI\n"
        logo += colorize("   /", 'blue')
        kwargs['description'] = logo
        return super().__init__(*args, **kwargs)

    def add_extra_args(self, args):
        sa = [a for a in self._actions if isinstance(a, argparse._SubParsersAction)]
        assert len(sa) == 1
        sa = sa[0]
        for _, v in sa.choices.items():
            v.add_extra_args(args)

    def parse_known_args(self, args=None, namespace=None, nohelp=False):
        known, unused = super().parse_known_args(args, namespace, nohelp)
        if hasattr(known, '_subparser'):
            # keep this around to keep the print message more in tune
            self._help_subparser = known._subparser
        return known, unused

    def print_help(self):
        """
        Print help, possibly deferring to the appropriate subcommand.
        """
        if self._help_subparser:
            self._help_subparser.print_help()
        else:
            return super().print_help()

    def add_subparsers(self, **kwargs):
        return super().add_subparsers(**kwargs)

    def _unsuppress_hidden(self):
        """
        Restore the help messages of hidden commands.
        """

        spa = [a for a in self._actions if isinstance(a, argparse._SubParsersAction)]
        assert len(spa) == 1
        spa = spa[0]
        for choices_action in spa._choices_actions:
            dest = choices_action.dest
            if choices_action.help == argparse.SUPPRESS:
                choices_action.help = spa.choices[dest].description

    def print_helpall(self):
        self._unsuppress_hidden()
        self.print_help()


class _SubcommandParser(ParlaiParser):
    """
    ParlaiParser which always sets add_parlai_args and add_model_args to False.

    Used in the superscript to initialize just the args for that command.
    """

    def __init__(self, **kwargs):
        kwargs['add_parlai_args'] = False
        kwargs['add_model_args'] = False
        assert 'description' in kwargs, 'Must supply description'
        return super().__init__(**kwargs)

    def parse_known_args(self, args=None, namespace=None, nohelp=False):
        if not nohelp:
            self.add_extra_args(args)
        return super().parse_known_args(args, namespace, nohelp)


def _SuperscriptHelpFormatter(**kwargs):
    kwargs['width'] = 100
    kwargs['max_help_position'] = 9999

    return CustomHelpFormatter(**kwargs)


def superscript_main(args=None):
    """
    Superscript is a loader for all the other scripts.
    """
    setup_script_registry()

    parser = _SupercommandParser(
        False, False, formatter_class=_SuperscriptHelpFormatter
    )
    parser.add_argument(
        '--helpall',
        action='helpall',
        help='List all commands, including advanced ones.',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=get_version_string(),
        help='Prints version info and exit.',
    )
    parser.set_defaults(super_command=None)
    subparsers = parser.add_subparsers(
        parser_class=_SubcommandParser, title="Commands", metavar="COMMAND"
    )
    hparser = subparsers.add_parser(
        'help',
        aliases=['h'],
        help=argparse.SUPPRESS,
        description='List the main commands.',
    )
    hparser.set_defaults(super_command='help')
    hparser = subparsers.add_parser(
        'helpall',
        help=argparse.SUPPRESS,
        description='List all commands, including advanced ones.',
    )
    hparser.set_defaults(super_command='helpall')

    # build the supercommand
    for script_name, registration in SCRIPT_REGISTRY.items():
        logging.verbose(f"Discovered command {script_name}")
        script_parser = registration.klass.setup_args()
        if script_parser is None:
            # user didn't bother defining command line args. let's just fill
            # in for them
            script_parser = ParlaiParser(False, False)
        help_ = argparse.SUPPRESS if registration.hidden else script_parser.description
        subparser = subparsers.add_parser(
            script_name,
            aliases=registration.aliases,
            help=help_,
            description=script_parser.description,
            formatter_class=CustomHelpFormatter,
        )
        subparser.set_defaults(
            # carries the name of the full command so we know what to execute
            super_command=script_name,
            # used in ParlAI parser to find CLI options set by user
            _subparser=subparser,
        )
        subparser.set_defaults(**script_parser._defaults)
        for action in script_parser._actions:
            subparser._add_action(action)
        for action_group in script_parser._action_groups:
            subparser._action_groups.append(action_group)

    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ModuleNotFoundError:
        pass

    opt = parser.parse_args(args)
    cmd = opt.pop('super_command')
    if cmd == 'helpall':
        parser.print_helpall()
    elif cmd == 'versioninfo':
        exit(0)
    elif cmd == 'help' or cmd is None:
        parser.print_help()
    elif cmd is not None:
        return SCRIPT_REGISTRY[cmd].klass._run_from_parser_and_opt(opt, parser)


def get_version_string() -> str:
    return f"ParlAI version {parlai.__version__}"
