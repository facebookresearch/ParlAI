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


if __name__ == '__main__':
    unittest.main()
