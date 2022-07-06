#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for exporting models via TorchScript (i.e. JIT compilation).

These do not require GPUs, but they are in nightly/gpu/ because they load fairseq, which
only the GPU CI checks install.
"""

import os
import regex
import unittest

import parlai.utils.testing as testing_utils
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.utils.bpe import Gpt2BpeHelper


# @testing_utils.skipUnlessFairseq
@testing_utils.skipUnlessTorch17
class TestTorchScript(unittest.TestCase):
    def test_gpt2_token_splitter(self):
        """
        Test TorchScriptable code for splitting tokens against reference GPT-2 version.
        """

        from parlai.scripts.torchscript import TorchScript
        from parlai.torchscript.tokenizer import ScriptableGpt2BpeHelper

        # Params
        tasks = ['taskmaster2', 'convai2']
        compiled_pattern = regex.compile(Gpt2BpeHelper.PATTERN)

        for task in tasks:
            opt = TorchScript.setup_args().parse_kwargs(
                task=task, datatype='train:ordered'
            )
            agent = RepeatLabelAgent(opt)
            # TODO(roller): make a proper create_teacher helper
            teacher = create_task(opt, agent).get_task_agent()
            num_examples = teacher.num_examples()

            print(
                f'\nStarting to test {num_examples:d} examples for the ' f'{task} task.'
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

    def test_subword_bpe_token_splitter(self):
        """
        Test TorchScriptable code for splitting tokens against reference subword BPE
        version.
        """

        from parlai.scripts.torchscript import TorchScript
        from parlai.torchscript.tokenizer import ScriptableSubwordBpeHelper

        # Params
        tasks = ['dialogue_safety:standard']
        splitter = regex.compile(r'\w+|[^\w\s]', regex.UNICODE)

        for task in tasks:
            opt = TorchScript.setup_args().parse_kwargs(
                task=task, datatype='train:ordered'
            )
            agent = RepeatLabelAgent(opt)
            teacher = create_task(opt, agent).get_task_agent()
            num_examples = teacher.num_examples()

            print(
                f'\nStarting to test {num_examples:d} examples for the ' f'{task} task.'
            )
            for idx, message in enumerate(teacher):
                if idx % 10000 == 0:
                    print(f'Testing example #{idx:d}.')
                text = message['text']
                canonical_tokens = splitter.findall(text)
                scriptable_tokens = ScriptableSubwordBpeHelper.findall(text)
                self.assertEqual(canonical_tokens, scriptable_tokens)
                if idx + 1 == num_examples:
                    break

    def test_special_tokenization(self):
        from parlai.core.dict import DictionaryAgent
        from parlai.core.params import ParlaiParser
        from parlai.torchscript.tokenizer import ScriptableDictionaryAgent

        SPECIAL = ['Q00', 'Q01']
        text = "Don't have a Q00, man! Have a Q01 instead."

        parser = ParlaiParser(False, False)
        DictionaryAgent.add_cmdline_args(parser)
        with testing_utils.tempdir() as tmp:
            opt = parser.parse_kwargs(
                dict_tokenizer='gpt2', dict_file=os.path.join(tmp, 'dict')
            )

            orig_dict = DictionaryAgent(opt)

            orig_bpe = orig_dict.bpe
            fused_key_bpe_ranks = {
                '\n'.join(key): float(val) for key, val in orig_bpe.bpe_ranks.items()
            }

            sda = ScriptableDictionaryAgent(
                null_token=orig_dict.null_token,
                end_token=orig_dict.end_token,
                unk_token=orig_dict.unk_token,
                start_token=orig_dict.start_token,
                freq=orig_dict.freq,
                tok2ind=orig_dict.tok2ind,
                ind2tok=orig_dict.ind2tok,
                bpe_add_prefix_space=False,
                bpe_encoder=orig_bpe.encoder,
                bpe_byte_encoder=orig_bpe.byte_encoder,
                fused_key_bpe_ranks=fused_key_bpe_ranks,
                special_tokens=[],
                subword_bpe_version=(0, 0),
                fused_bpe_codes={},
                subword_bpe_separator='',
            )

            tokenized = sda.txt2vec(text, dict_tokenizer='gpt2')
            assert len(tokenized) == 15
            assert sda.vec2txt(tokenized, dict_tokenizer='gpt2') == text

            orig_dict = DictionaryAgent(opt)
            orig_dict.add_additional_special_tokens(SPECIAL)
            orig_bpe = orig_dict.bpe
            sda = ScriptableDictionaryAgent(
                null_token=orig_dict.null_token,
                end_token=orig_dict.end_token,
                unk_token=orig_dict.unk_token,
                start_token=orig_dict.start_token,
                freq=orig_dict.freq,
                tok2ind=orig_dict.tok2ind,
                ind2tok=orig_dict.ind2tok,
                bpe_add_prefix_space=False,
                bpe_encoder=orig_bpe.encoder,
                bpe_byte_encoder=orig_bpe.byte_encoder,
                fused_key_bpe_ranks=fused_key_bpe_ranks,
                special_tokens=SPECIAL,
                subword_bpe_version=(0, 0),
                fused_bpe_codes={},
                subword_bpe_separator='',
            )

            special_tokenized = sda.txt2vec(text, dict_tokenizer='gpt2')
            assert len(special_tokenized) == 15
            assert sda.vec2txt(special_tokenized, dict_tokenizer='gpt2') == text
            assert special_tokenized != tokenized

    def test_torchscript_bart_agent(self):
        """
        Test exporting a BART model to TorchScript and then testing it on sample data.
        """

        from parlai.scripts.torchscript import TorchScript

        test_phrase = "Don't have a cow, man!"  # From test_bart.py

        with testing_utils.tempdir() as tmpdir:

            scripted_model_file = os.path.join(tmpdir, 'scripted_model.pt')

            # Export the BART model
            export_opt = TorchScript.setup_args().parse_kwargs(
                model='bart', scripted_model_file=scripted_model_file, no_cuda=True
            )
            TorchScript(export_opt).run()

            # Test the scripted BART model
            scripted_opt = ParlaiParser(True, True).parse_kwargs(
                model='parlai.torchscript.agents:TorchScriptAgent',
                model_file=scripted_model_file,
            )
            bart = create_agent(scripted_opt)
            bart.observe({'text': test_phrase, 'episode_done': True})
            act = bart.act()
            self.assertEqual(act['text'], test_phrase)

    def test_torchscript_transformer_classifier_agent(self):
        """
        Test exporting a Transformer classifier model to TorchScript and then testing it
        on sample data.
        """

        from parlai.scripts.torchscript import TorchScript

        test_phrase = "Don't have a cow, man!"

        with testing_utils.tempdir() as tmpdir:

            scripted_model_file = os.path.join(tmpdir, 'scripted_model.pt')

            # Export transformer classifier model for safety task
            export_opt = TorchScript.setup_args().parse_kwargs(
                model='transformer/classifier',
                model_file='zoo:dialogue_safety/single_turn/model',
                script_module='parlai.torchscript.modules:TorchScriptTransformerClassifier',
                scripted_model_file=scripted_model_file,
                no_cuda=True,
            )
            TorchScript(export_opt).run()

            # Test the scripted transformer classifier model
            scripted_opt = ParlaiParser(True, True).parse_kwargs(
                model='parlai.torchscript.agents:TorchScriptAgent',
                model_file=scripted_model_file,
            )
            bart = create_agent(scripted_opt)
            bart.observe({'text': test_phrase, 'episode_done': True})
            act = bart.act()
            self.assertEqual(act['text'], '__ok__')

    def test_gpu_torchscript_bart_agent(self):
        """
        Test exporting a BART model to TorchScript for GPU and then testing it on sample
        data.
        """

        from parlai.scripts.torchscript import TorchScript

        test_phrase = "Don't have a cow, man!"  # From test_bart.py

        with testing_utils.tempdir() as tmpdir:

            scripted_model_file = os.path.join(tmpdir, 'scripted_model.pt')

            # Export the BART model for GPU
            export_opt = TorchScript.setup_args().parse_kwargs(
                model='bart', scripted_model_file=scripted_model_file, no_cuda=False
            )
            TorchScript(export_opt).run()

            # Test the scripted GPU BART model
            scripted_opt = ParlaiParser(True, True).parse_kwargs(
                model='parlai.torchscript.agents:TorchScriptAgent',
                model_file=scripted_model_file,
            )
            bart = create_agent(scripted_opt)
            bart.observe({'text': test_phrase, 'episode_done': True})
            act = bart.act()
            self.assertEqual(act['text'], test_phrase)


if __name__ == '__main__':
    unittest.main()
