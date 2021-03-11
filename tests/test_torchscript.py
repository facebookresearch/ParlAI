#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for exporting models via TorchScript.
"""

import regex
import unittest

from parlai.torchscript.export_model import ScriptableGpt2BpeHelper
from parlai.utils.bpe import Gpt2BpeHelper


class TestTorchscript(unittest.TestCase):
    def test_token_splitter(self):
        """
        Test TorchScriptable code for splitting tokens against reference GPT-2 version.
        """
        test_strings = [
            " You 3, it's 2021, I'll be doing that   2morrow!  ",
            'How to tokenize for GPT2?!',
            'I love parfaits with açaí ',
            "He's't're've",
            '  \t\t \t5 \t',
            '\t\t  \t 5\t ',
        ]
        # TODO: just loop through all of the BST val set, for instance, to get more cases?

        compiled_pattern = regex.compile(Gpt2BpeHelper.PATTERN)

        for str_ in test_strings:
            canonical_tokens = regex.findall(compiled_pattern, str_)
            scriptable_tokens = ScriptableGpt2BpeHelper.findall(str_)
            self.assertEqual(canonical_tokens, scriptable_tokens)


if __name__ == '__main__':
    unittest.main()
