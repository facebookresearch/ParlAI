#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for exporting models via TorchScript.
"""

import regex
import unittest

import parlai.utils.testing as testing_utils
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.scripts.display_data import setup_args
from parlai.torchscript.export_model import ScriptableGpt2BpeHelper
from parlai.utils.bpe import Gpt2BpeHelper


class TestTorchscript(unittest.TestCase):
    def test_token_splitter(self):
        """
        Test TorchScriptable code for splitting tokens against reference GPT-2 version.
        """

        # Params
        tasks = ['taskmaster2', 'convai2']
        compiled_pattern = regex.compile(Gpt2BpeHelper.PATTERN)

        with testing_utils.tempdir() as tmpdir:
            datapath = tmpdir

            for task in tasks:
                parser = setup_args()
                args = f"""\
--task {task}
--datatype train:ordered
--datapath {datapath}
"""
                opt = parser.parse_args(args.split())
                agent = RepeatLabelAgent(opt)
                teacher = create_task(opt, agent).get_task_agent()
                num_examples = teacher.num_examples()

                print(
                    f'\nStarting to test {num_examples:d} examples for the '
                    f'{task} task.'
                )
                for idx, message in enumerate(teacher):
                    if idx % 10000 == 0:
                        print(f'Testing example #{idx:d}.')
                    text = message['text']
                    canonical_tokens = regex.findall(compiled_pattern, text)
                    scriptable_tokens = ScriptableGpt2BpeHelper.findall(text)
                    self.assertEqual(canonical_tokens, scriptable_tokens)
                    if idx + 1 == num_examples:
                        break


if __name__ == '__main__':
    unittest.main()
