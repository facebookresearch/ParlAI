#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for exporting models via TorchScript (i.e. JIT compilation).

These do not require GPUs, but they are in nightly/gpu/ because they load fairseq, which
only the GPU CI checks install.
"""

import unittest

import parlai.utils.testing as testing_utils


@testing_utils.skipUnlessFairseq
@testing_utils.skipUnlessTorch17
class TestTorchScript(unittest.TestCase):
    def test_token_splitter(self):
        """
        Test TorchScriptable code for splitting tokens against reference GPT-2 version.
        """

        pass

    def test_torchscript_agent(self):
        """
        Test exporting a model to TorchScript and then testing it on sample data.
        """

        pass


if __name__ == '__main__':
    unittest.main()
