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


if __name__ == '__main__':
    unittest.main()
