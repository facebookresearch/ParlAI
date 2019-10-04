#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test ParlaiParser and other opt/params.py code."""

import os
import json
import unittest
from parlai.core.params import ParlaiParser
import parlai.core.agents as agents
import parlai.core.testing_utils as testing_utils


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
    """Test ParlaiParser."""

    def test_upgrade_opt(self):
        """Test whether upgrade_opt works."""
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


if __name__ == '__main__':
    unittest.main()
