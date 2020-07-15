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
from typing import List, Optional, Dict, Any
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from abc import abstractmethod
import importlib
import pkgutil
import parlai.scripts
from parlai.core.loader import register_script, SCRIPT_REGISTRY


def setup_script_registry():
    """
    Loads the scripts so that @register_script is hit for all.
    """
    for module in pkgutil.iter_modules(parlai.scripts.__path__, 'parlai.scripts.'):
        if module.name == 'parlai.scripts.parlai_exe':
            continue
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
    logo = colorize('ParlAI - Dialogue Research Platform', 'labels')
    print(logo)


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


def superscript_main(args=None):
    """
    Superscript is a loader for all the other scripts.
    """

    setup_script_registry()
    parser = ParlaiParser(False, False, description='ParlAI Root launcher')
    subparsers = parser.add_subparsers(
        dest='cmd',
        help='ParlAI Commands',
        required=True,
        parser_class=_SubcommandParser,
    )

    for script_name, klass in SCRIPT_REGISTRY.items():
        script_parser = klass.setup_args()
        subparser = subparsers.add_parser(script_name, help=script_parser.description)
        for action in script_parser._actions:
            subparser._add_action(action)

    opt = parser.parse_args(args)
    cmd = opt.pop('cmd')
    SCRIPT_REGISTRY[cmd](opt).run()
