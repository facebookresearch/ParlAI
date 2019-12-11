#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test ParlaiParser and other opt/params.py code.
"""

import os
import re
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
            with testing_utils.capture_output() as _:
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
        parser = ParlaiParser()
        parser.add_argument(
            '-bs',
            '--batchsize',
            default=1,
            type=int,
            help='batch size for minibatch training schemes',
            recommended="10",
        )

        with testing_utils.capture_output() as _:
            parser.parse_args()
        help_str = parser.format_help()
        assert re.search(r'--batchsize[^\n]*\n[^\n]*\(recommended: 10\)', help_str)

    def test_recommendations_group(self):
        """
        Test whether recommended args work for a group.
        """
        parser = ParlaiParser()
        parser_grp = parser.add_argument_group('Test Group')
        parser_grp.add_argument(
            '-bs',
            '--batchsize',
            default=1,
            type=int,
            help='batch size for minibatch training schemes',
            recommended=[5, 10, 15],
        )
        with testing_utils.capture_output() as _:
            parser.parse_args()

        help_str = parser.format_help()
        assert re.search(r'Test Group:\n', help_str)
        assert re.search(
            r'--batchsize[^\n]*\n[^\n]*\(recommended: \[5, 10, 15\]\)', help_str
        )


if __name__ == '__main__':
    unittest.main()
