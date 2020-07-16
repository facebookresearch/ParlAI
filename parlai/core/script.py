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

import os
import io
import argparse
from typing import List, Optional, Dict, Any
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser, CustomHelpFormatter
from abc import abstractmethod
import importlib
import pkgutil
import parlai.scripts
from parlai.core.loader import register_script, SCRIPT_REGISTRY  # noqa: F401


def setup_script_registry():
    """
    Loads the scripts so that @register_script is hit for all.
    """
    for module in pkgutil.iter_modules(parlai.scripts.__path__, 'parlai.scripts.'):
        importlib.import_module(module.name)


class ParlaiScript(object):
    """
    A ParlAI script is a standardized form of command line args.
    """

    description: str = "Default Script Description"
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
        pass

    @classmethod
    def _run_kwargs(cls, kwargs: Dict[str, Any]):
        """
        Construct and run the script using kwargs, pseudo-parsing them.
        """
        parser = cls.setup_args()
        opt = parser.parse_kwargs(**kwargs)
        script = cls(opt)
        script.parser = parser
        return script.run()

    @classmethod
    def _run_args(cls, args: Optional[List[str]] = None):
        """
        Construct and run the script using args, defaulting to getting from CLI.
        """
        parser = cls.setup_args()
        opt = parser.parse_args(args=args, print_args=False)
        script = cls(opt)
        script.parser = parser
        return script.run()

    @classmethod
    def main(cls, *args, **kwargs):
        """
        Run the program.
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


def _display_image():
    if os.environ.get('PARLAI_DISPLAY_LOGO') == 'OFF':
        return
    from parlai.utils.strings import colorize

    logo = colorize('ParlAI - Dialogue Research Platform', 'labels')
    print(logo)


class _SupercommandParser(ParlaiParser):
    def add_extra_args(self, args):
        sa = [a for a in self._actions if isinstance(a, argparse._SubParsersAction)]
        assert len(sa) == 1
        sa = sa[0]
        for _, v in sa.choices.items():
            v.add_extra_args(args)


class _SubcommandParser(ParlaiParser):
    """
    ParlaiParser which always sets add_parlai_args and add_model_args to False.

    Used in the superscript to initialize just the args for that command.
    """

    def __init__(self, **kwargs):
        kwargs['add_parlai_args'] = False
        kwargs['add_model_args'] = False
        if 'description' not in kwargs:
            kwargs['description'] = None
        return super().__init__(**kwargs)

    def parse_known_args(self, args=None, namespace=None, nohelp=False):
        if not nohelp:
            self.add_extra_args(args)
        return super().parse_known_args(args, namespace, nohelp)


def _CustomHelpFormatter(**kwargs):
    kwargs['width'] = None
    kwargs['max_help_position'] = 9999
    return CustomHelpFormatter(**kwargs)


def superscript_main(args=None):
    """
    Superscript is a loader for all the other scripts.
    """

    setup_script_registry()

    parser = _SupercommandParser(False, False, formatter_class=_CustomHelpFormatter)
    subparsers = parser.add_subparsers(
        parser_class=_SubcommandParser, title="Commands", metavar="COMMAND",
    )
    subparsers.add_parser('help', help=argparse.SUPPRESS, aliases=['h'])

    for script_name, registration in SCRIPT_REGISTRY.items():
        script_parser = registration.klass.setup_args()
        is_hidden = registration.hidden
        is_hidden = is_hidden or script_parser.description is None
        help_ = argparse.SUPPRESS if is_hidden else script_parser.description
        subparser = subparsers.add_parser(
            script_name,
            aliases=registration.aliases,
            help=help_,
            formatter_class=CustomHelpFormatter,
        )
        subparser.set_defaults(super_command=script_name)
        for action in script_parser._actions:
            subparser._add_action(action)

    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ModuleNotFoundError:
        pass

    opt = parser.parse_args(args, print_args=False)
    cmd = opt.pop('super_command')
    if cmd == 'help' or cmd is None:
        _display_image()
        parser.print_help()
    elif cmd is not None:
        SCRIPT_REGISTRY[cmd].klass(opt).run()
