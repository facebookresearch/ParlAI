#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test ParlaiParser and other opt/params.py code.
"""

import os
import json
import unittest
from parlai.core.params import ParlaiParser
import parlai.core.agents as agents
import parlai.utils.testing as testing_utils


class _ExampleUpgradeOptAgent(agents.Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        assert 'is_upgraded' in opt
        assert opt['is_upgraded'] is True

    @classmethod
    def upgrade_opt(cls, opt):
        opt = super(_ExampleUpgradeOptAgent, cls).upgrade_opt(opt)
        assert 'is_upgraded' not in opt
        opt['is_upgraded'] = True
        return opt


class TestParlaiParser(unittest.TestCase):
    """
    Test ParlaiParser.
    """

    def test_shortopt(self):
        """
        Tests whether short opts like -mtw work.

        Known to be tricky in python 3.8.
        """
        pp = ParlaiParser(False, False)
        pp.add_argument("-m", "--model")
        pp.add_argument("-mtw", "--multitask-weights")
        opt = pp.parse_args(["-m", "memnn"])
        print(opt)

    def test_opt_presets(self):
        """
        Tests whether opt presets bundled with parlai work as expected.
        """
        pp = ParlaiParser(True, False)
        pp.add_argument("-m", "--model")
        # hardcoded example
        opt = pp.parse_args(['--model', 'transformer/generator', '-o', 'gen/meena'])
        assert opt['beam_size'] == 20
        assert opt['inference'] == 'topk'
        assert opt['topk'] == 40
        # and preference for command line over opt presets
        pp = ParlaiParser(True, False)
        pp.add_argument("-m", "--model")
        opt = pp.parse_args(
            ['--model', 'transformer/generator', '-o', 'gen/meena', '--topk', '7']
        )
        assert opt['beam_size'] == 20
        assert opt['inference'] == 'topk'
        assert opt['topk'] == 7
        # double check ordering doesn't matter
        pp = ParlaiParser(True, False)
        pp.add_argument("-m", "--model")
        opt = pp.parse_args(
            ['--model', 'transformer/generator', '--topk', '8', '-o', 'gen/meena']
        )
        assert opt['beam_size'] == 20
        assert opt['inference'] == 'topk'
        assert opt['topk'] == 8
        # check composability
        pp = ParlaiParser(True, False)
        pp.add_argument("-m", "--model")
        opt = pp.parse_args(['-o', 'arch/blenderbot_3B,gen/meena'])
        assert opt['beam_size'] == 20
        assert opt['inference'] == 'topk'
        assert opt['model'] == 'transformer/generator'
        assert opt['n_encoder_layers'] == 2

    def test_upgrade_opt(self):
        """
        Test whether upgrade_opt works.
        """
        with testing_utils.tempdir() as tmp:
            modfn = os.path.join(tmp, 'model')
            with open(modfn, 'w') as f:
                f.write('Test.')
            optfn = modfn + '.opt'
            base_opt = {
                'model': 'tests.test_params:_ExampleUpgradeOptAgent',
                'dict_file': modfn + '.dict',
                'model_file': modfn,
            }
            with open(optfn, 'w') as f:
                json.dump(base_opt, f)

            pp = ParlaiParser(True, True)
            opt = pp.parse_args(['--model-file', modfn])
            agents.create_agent(opt)

    def test_recommendations_single(self):
        """
        Test whether recommended args work for non-group.
        """
        parser = ParlaiParser(False, False)
        parser.add_argument(
            '-bs',
            '--batchsize',
            default=1,
            type=int,
            help='batch size for minibatch training schemes',
            recommended=1337,
        )
        parser.parse_args([])
        help_str = parser.format_help()
        assert 'recommended:' in help_str
        assert '1337' in help_str

    def test_recommendations_group(self):
        """
        Test whether recommended args work for a group.
        """
        parser = ParlaiParser(False, False)
        parser_grp = parser.add_argument_group('Test Group')
        parser_grp.add_argument(
            '-bs',
            '--batchsize',
            default=1,
            type=int,
            help='batch size for minibatch training schemes',
            recommended=1337,
        )
        parser.parse_args([])

        help_str = parser.format_help()
        assert 'Test Group' in help_str
        _, latter = help_str.split('Test Group')
        assert 'recommended:' in latter
        assert '1337' in latter

    def test_parse_kwargs(self):
        parser = ParlaiParser(True, True)

        # implied args from the model
        opt = parser.parse_kwargs(model='transformer/generator', relu_dropout=0.3)
        assert opt['relu_dropout'] == 0.3
        assert opt['model'] == 'transformer/generator'
        assert 'n_heads' in opt

        # bad types
        with self.assertRaises(ValueError):
            parser = ParlaiParser(True, True)
            parser.parse_kwargs(model='transformer/generator', relu_dropout='foo')

        # nonexistant args without model
        with self.assertRaises(KeyError):
            parser = ParlaiParser(True, True)
            parser.parse_kwargs(fake_arg='foo')

        # nonexistant args with model
        with self.assertRaises(KeyError):
            parser = ParlaiParser(True, True)
            parser.parse_kwargs(model='transformer/generator', fake_arg='foo')

    def test_parse_kwargs_multirounds(self):
        """
        Test parse_kwargs when we have options that depend on options.
        """
        parser = ParlaiParser(True, False)
        opt = parser.parse_kwargs(
            task='integration_tests', mutators='episode_shuffle', preserve_context=True
        )
        assert opt['preserve_context'] is True
        opt = parser.parse_kwargs(
            task='integration_tests', mutators='episode_shuffle', preserve_context=False
        )
        assert opt['preserve_context'] is False

        with self.assertRaises(KeyError):
            parser.parse_kwargs(
                task='integration_tests', mutators='episode_shuffle', fake_option=False
            )

        with self.assertRaises(KeyError):
            parser.parse_kwargs(task='integration_tests', fake_option=False)

    def test_parse_kwargs_nargsplus(self):
        """
        Test parse_kwargs when provided an argument with >1 item.
        """
        parser = ParlaiParser(False, False)
        parser.add_argument('--example', nargs='+', choices=['a', 'b', 'c'])
        opt = parser.parse_args(['--example', 'a', 'b'])
        assert opt['example'] == ['a', 'b']

        parser = ParlaiParser(False, False)
        parser.add_argument('--example', nargs='+', choices=['a', 'b', 'c'])
        opt = parser.parse_kwargs(example=['a', 'b'])
        assert opt['example'] == ['a', 'b']

        parser = ParlaiParser(False, False)
        parser.add_argument('--example', nargs='+')
        opt = parser.parse_kwargs(example=['x', 'y'])
        assert opt['example'] == ['x', 'y']

    def test_bool(self):
        """
        test add_argument(type=bool)
        """
        parser = ParlaiParser(True, True)
        parser.add_argument('--foo', type=bool)
        opt = parser.parse_args(['--foo', 'true'])
        assert opt['foo'] is True
        opt = parser.parse_args(['--foo', 'False'])
        assert opt['foo'] is False
        opt = parser.parse_args(['--foo', '0'])
        assert opt['foo'] is False

        group = parser.add_argument_group('foo container')
        group.add_argument('--bar', type=bool)
        opt = parser.parse_args(['--bar', 'true'])
        assert opt['bar'] is True
        opt = parser.parse_args(['--bar', 'False'])
        assert opt['bar'] is False
        opt = parser.parse_args(['--bar', '0'])
        assert opt['bar'] is False

        parser = ParlaiParser(True, True)
        parser.add_argument('--foo', type='bool')
        opt = parser.parse_args(['--foo', 'true'])
        assert opt['foo'] is True
        opt = parser.parse_args(['--foo', 'False'])
        assert opt['foo'] is False
        opt = parser.parse_args(['--foo', '0'])
        assert opt['foo'] is False

        group = parser.add_argument_group('foo container')
        group.add_argument('--bar', type='bool')
        opt = parser.parse_args(['--bar', 'true'])
        assert opt['bar'] is True
        opt = parser.parse_args(['--bar', 'False'])
        assert opt['bar'] is False
        opt = parser.parse_args(['--bar', '0'])
        assert opt['bar'] is False


class TestAddCmdlineArgs(unittest.TestCase):
    """
    Test that a deprecated api signature for add_cmdline_args is not accepted.
    """

    def test_bad_agent(self):
        from parlai.core.agents import register_agent

        @register_agent("bad_addcmdlineargs_agent")
        class TestAgent:
            @classmethod
            def add_cmdline_args(cls, argparser):
                argparser.add_argument("--no-find", default=True)
                return argparser

        with self.assertRaises(TypeError) as cm:
            ParlaiParser(True, True).parse_kwargs(
                model="bad_addcmdlineargs_agent", task="integration_tests"
            )
            self.assertIn("add_cmdline_args(argparser)", "\n".join(cm.output))

    def test_good_agent(self):
        from parlai.core.agents import register_agent

        @register_agent("partialopt_addcmdlineargs_agent")
        class TestAgent:
            @classmethod
            def add_cmdline_args(cls, argparser, partial_opt=None):
                argparser.add_argument("--yes-find", default=True, type='bool')
                if partial_opt and partial_opt.get('yes_find'):
                    # conditional argument addition
                    argparser.add_argument(
                        "--yes-partial-find", default=True, type='bool'
                    )
                return argparser

        opt = ParlaiParser(True, True).parse_kwargs(
            model="partialopt_addcmdlineargs_agent", task="integration_tests"
        )
        assert 'yes_find' in opt
        assert 'yes_partial_find' in opt

        # don't trigger the partial opt dynamic add. we shouldn't find the subarg.
        opt = ParlaiParser(True, True).parse_kwargs(
            model="partialopt_addcmdlineargs_agent",
            task="integration_tests",
            yes_find=False,
        )
        assert 'yes_find' in opt
        assert 'yes_partial_find' not in opt

    def test_bad_task(self):
        from parlai.core.teachers import register_teacher

        @register_teacher("bad_addcmdlineargs_teacher")
        class TestAgent:
            @classmethod
            def add_cmdline_args(cls, argparser):
                argparser.add_argument("--no-find", default=True)
                return argparser

        with self.assertRaises(TypeError) as cm:
            ParlaiParser(True, True).parse_kwargs(
                model="repeat_query", task="bad_addcmdlineargs_teacher"
            )
            self.assertIn("add_cmdline_args(argparser)", "\n".join(cm.output))

    def test_good_task(self):
        from parlai.core.teachers import register_teacher

        @register_teacher("partialopt_addcmdlineargs_teacher")
        class TestAgent:
            @classmethod
            def add_cmdline_args(cls, argparser, partial_opt=None):
                argparser.add_argument("--yes-find", default=True, type='bool')
                if partial_opt and partial_opt.get('yes_find'):
                    # conditional argument addition
                    argparser.add_argument(
                        "--yes-partial-find", default=True, type='bool'
                    )
                return argparser

        opt = ParlaiParser(True, True).parse_kwargs(
            model="repeat_query", task="partialopt_addcmdlineargs_teacher"
        )
        assert 'yes_find' in opt
        assert 'yes_partial_find' in opt

        # don't trigger the partial opt dynamic add. we shouldn't find the subarg.
        opt = ParlaiParser(True, True).parse_kwargs(
            model="partialopt_addcmdlineargs_agent",
            task="integration_tests",
            yes_find=False,
        )
        assert 'yes_find' in opt
        assert 'yes_partial_find' not in opt


if __name__ == '__main__':
    unittest.main()
