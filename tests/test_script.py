#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
from unittest.mock import patch
from parlai.core.params import ParlaiParser
import parlai.core.script as script
import parlai.utils.testing as testing_utils


@script.register_script("test_script")
class _TestScript(script.ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(True, False, description='My Description')
        parser.add_argument('--foo', default='defaultvalue')
        parser.add_argument('--bar', default='sneaky', hidden=True)
        return parser

    def run(self):
        return self.opt


@script.register_script("hidden_script", hidden=True)
class _HiddenScript(_TestScript):
    pass


@script.register_script("no_setup_args")
class _NoSetupArgsScript(script.ParlaiScript):
    pass


class TestScriptRegistry(unittest.TestCase):
    def test_setup_script(self):
        script.setup_script_registry()
        assert 'train_model' in script.SCRIPT_REGISTRY

    def test_register_script(self):
        assert 'test_script' in script.SCRIPT_REGISTRY

    def test_main_kwargs(self):
        opt = _TestScript.main(foo='test')
        assert opt.get('foo') == 'test'
        assert opt.get('bar') == 'sneaky'

    def test_main_args(self):
        opt = _TestScript.main('--foo', 'test')
        assert opt.get('foo') == 'test'
        assert opt.get('bar') == 'sneaky'

    def test_main_noargs(self):
        with patch.object(sys, 'argv', ['test_script.py']):  # argv[0] doesn't matter
            opt = _TestScript.main()
        assert opt.get('foo') == 'defaultvalue'
        assert opt.get('bar') == 'sneaky'

    def test_help(self):
        helptext = _TestScript.help()
        assert 'My Description' in helptext
        assert '--foo' in helptext
        assert '--bar' not in helptext

        with testing_utils.capture_output() as output:
            with self.assertRaises(SystemExit):
                _TestScript.main('--help')
            assert '--foo' in output.getvalue()
            assert '--bar' not in output.getvalue()

        with testing_utils.capture_output() as output:
            with self.assertRaises(SystemExit):
                _TestScript.main('--helpall')
            assert '--foo' in output.getvalue()
            assert '--bar' in output.getvalue()


class TestSuperCommand(unittest.TestCase):
    def test_supercommand(self):
        opt = script.superscript_main(args=['test_script', '--foo', 'test'])
        assert opt.get('foo') == 'test'

    def test_no_setup_args(self):
        with self.assertRaises(NotImplementedError):
            script.superscript_main(args=['no_setup_args'])

    def test_help(self):
        with testing_utils.capture_output() as output:
            script.superscript_main(args=['help'])
            assert 'test_script' in output.getvalue()
            assert 'hidden_script' not in output.getvalue()
            # showing help for the super command, not the subcommand
            assert '--foo' not in output.getvalue()

        with testing_utils.capture_output() as output:
            script.superscript_main(args=['helpall'])
            assert 'test_script' in output.getvalue()
            assert 'hidden_script' in output.getvalue()

    def test_subcommand_help(self):
        with testing_utils.capture_output() as output:
            with self.assertRaises(SystemExit):
                script.superscript_main(args=['test_script', 'foo'])
                assert 'parlai test_script' in output.getvalue()
                assert '--foo' in output.getvalue()
