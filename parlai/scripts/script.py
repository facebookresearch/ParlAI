#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
The general ParlAI Script interface.

An abstract class to help standardize the call to ParlAI scripts, enabling them to
be completed easily.
"""

import io
from typing import List, Optional, Dict, Any
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from abc import abstractmethod


class ParlaiScript(object):
    """
    A ParlAI script is a standardized form of command line args.
    """

    description: str = "Default Script Description"

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
        return cls(opt).run()

    @classmethod
    def _run_args(cls, args: Optional[List[str]] = None):
        """
        Construct and run the script using args, defaulting to getting from CLI.
        """
        parser = cls.setup_args()
        opt = parser.parse_args(args=args, print_args=False)
        return cls(opt).run()

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
    def help(cls, *args, **kwargs):
        f = io.StringIO()
        cls.setup_args().print_help(f)
        return f.getvalue()
