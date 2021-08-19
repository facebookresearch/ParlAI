#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dictionary testing.
"""

from parlai.core.build_data import modelzoo_path
from parlai.core.dict import find_ngrams
from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent, TokenizationMode
from parlai.core.opt import Opt
import parlai.scripts.build_dict as build_dict

from parlai.utils.io import PathManager
import parlai.utils.testing as testing_utils
import os
import shutil
import unittest
import pytest

try:
    from tokenizers import ByteLevelBPETokenizer  # @manual # noqa: F401

    TOKENIZERS = True
except ImportError:
    TOKENIZERS = False

DEFAULT_BYTELEVEL_BPE_VOCAB = (
    'zoo:unittest/test_bytelevel_bpe_v2/test-byte-level-bpe-v2-vocab.json'
)
DEFAULT_BYTELEVEL_BPE_MERGE = (
    'zoo:unittest/test_bytelevel_bpe_v2/test-byte-level-bpe-v2-merges.txt'
)
BYTELEVEL_BPE_RESULT = [
    'H',
    'ello',
    ',',
    'Ġ',
    'P',
    'ar',
    'l',
    'A',
    'I',
    '!',
    'Ġ',
    'ð',
    'Ł',
    'ĺ',
    'Ģ',
]
GPT2_BPE_RESULT = [
    'Hello',
    ',',
    r'\xc4\xa0Par',
    'l',
    'AI',
    '!',
    r'\xc4\xa0\xc3\xb0\xc5\x81\xc4\xba',
    r'\xc4\xa2',
]
slow_bytelevel_bpe_RESULT = [
    'H',
    'ello',
    ',',
    '\\xc4\\xa0',
    'P',
    'ar',
    'l',
    'A',
    'I',
    '!',
    '\\xc4\\xa0',
    '\\xc3\\xb0',
    '\\xc5\\x81',
    '\\xc4\\xba',
    '\\xc4\\xa2',
]


class TestDictionary(unittest.TestCase):
    """
    Basic tests on the built-in parlai Dictionary.
    """

    def test_gpt2_bpe_tokenize(self):
        datapath = ParlaiParser().parse_args([], print_args=False)['datapath']
        opt = Opt({'dict_tokenizer': 'gpt2', 'datapath': datapath})
        agent = DictionaryAgent(opt)
        self.assertEqual(
            # grinning face emoji
            agent.gpt2_tokenize(u'Hello, ParlAI! \U0001f600'),
            GPT2_BPE_RESULT,
        )
        self.assertEqual(
            agent.vec2txt(agent.tok2ind[w] for w in GPT2_BPE_RESULT),
            # grinning face emoji
            u'Hello, ParlAI! \U0001f600',
        )

    def test_space_tokenize(self):
        """
        Space tokenize assumes raw tokenization as input.
        """
        self.assertEqual(
            DictionaryAgent.space_tokenize('   this is a test!   '),
            ['this', 'is', 'a', 'test!'],
        )

    def test_split_tokenize(self):
        """
        Split tokenize specially handles some limited punctuation.
        """
        self.assertEqual(
            DictionaryAgent.split_tokenize('   this is a test!   '),
            ['this', 'is', 'a', 'test', '!'],
        )

    def test_find_ngrams(self):
        """
        Test the ngram class properly recognize uni, bi, and trigrams test.
        """
        s = set()
        s.add('hello world')
        s.add('ol boy')
        res = find_ngrams(s, ['hello', 'world', 'buddy', 'ol', 'boy'], 2)
        assert ' '.join(res) == 'hello world buddy ol boy'
        assert '-'.join(res) == 'hello world-buddy-ol boy'
        s.add('world buddy ol')
        res = find_ngrams(s, ['hello', 'world', 'buddy', 'ol', 'boy'], 3)
        assert ' '.join(res) == 'hello world buddy ol boy'
        assert '-'.join(res) == 'hello-world buddy ol-boy'
        s.add('hello world buddy')
        res = find_ngrams(s, ['hello', 'world', 'buddy', 'ol', 'boy'], 3)
        assert ' '.join(res) == 'hello world buddy ol boy'
        assert '-'.join(res) == 'hello world buddy-ol boy'

    def test_basic_parse(self):
        """
        Check the dictionary is correctly adding and parsing short sentence.
        """
        parser = ParlaiParser()
        DictionaryAgent.add_cmdline_args(parser, partial_opt=None)
        opt = parser.parse_args([])
        dictionary = DictionaryAgent(opt)
        num_builtin = len(dictionary)

        dictionary.observe({'text': 'hello world'})
        dictionary.act()
        assert len(dictionary) - num_builtin == 2

        vec = dictionary.parse('hello world')
        assert len(vec) == 2
        assert vec[0] == num_builtin
        assert vec[1] == num_builtin + 1

        vec = dictionary.parse('hello world', vec_type=list)
        assert len(vec) == 2
        assert vec[0] == num_builtin
        assert vec[1] == num_builtin + 1

        vec = dictionary.parse('hello world', vec_type=tuple)
        assert len(vec) == 2
        assert vec[0] == num_builtin
        assert vec[1] == num_builtin + 1

    @pytest.mark.nofbcode
    def test_set_model_file_without_dict_file(self):
        """
        Check that moving a model without moving the dictfile raises an error.
        """
        # Download model, move to a new location
        with testing_utils.tempdir() as datapath:
            try:
                # remove unittest models if there before
                shutil.rmtree(os.path.join(datapath, 'models/unittest'))
            except FileNotFoundError:
                pass

            zoo_path = 'zoo:unittest/seq2seq/model'
            model_path = modelzoo_path(datapath, zoo_path)
            PathManager.rm(model_path + '.dict')
            # Test that eval model fails
            with self.assertRaises(RuntimeError):
                testing_utils.eval_model(
                    dict(task='babi:task1k:1', model_file=model_path)
                )
            try:
                # remove unittest models if there after
                shutil.rmtree(os.path.join(datapath, 'models/unittest'))
            except FileNotFoundError:
                pass

    def test_train_model_with_no_dict_file(self):
        """
        Ensure training a model requires a dict_file or model_file.
        """
        import parlai.scripts.train_model as tms

        parser = tms.setup_args()
        parser.set_params(task='babi:task1k:1', model='seq2seq')
        popt = parser.parse_args([])
        with self.assertRaises(RuntimeError):
            tms.TrainLoop(popt)


@unittest.skipUnless(TOKENIZERS, "No tokenizers available")
class TestByteLevelBPE(unittest.TestCase):
    """
    Test ByteLevelBPE is well-behaved.
    """

    def test_tokenize_prefix_space(self):
        """
        Tests a bytelevel bpe tokenizer inside ParlAI.
        """
        parser = ParlaiParser()
        parser.set_params(
            dict_tokenizer='bytelevelbpe',
            bpe_vocab=DEFAULT_BYTELEVEL_BPE_VOCAB,
            bpe_merge=DEFAULT_BYTELEVEL_BPE_MERGE,
        )
        opt = parser.parse_args([])
        agent = DictionaryAgent(opt)
        self.assertEqual(
            # grinning face emoji
            agent.bytelevelbpe_tokenize(u'Hello, ParlAI! \U0001f600'),
            ['Ġ'] + BYTELEVEL_BPE_RESULT,
        )
        self.assertEqual(
            agent.vec2txt([agent.tok2ind[w] for w in ['Ġ'] + BYTELEVEL_BPE_RESULT]),
            # grinning face emoji
            u'Hello, ParlAI! \U0001f600',
        )
        self.assertEqual(
            agent.txt2vec(u'Hello, ParlAI! \U0001f600'),
            [agent.tok2ind[w] for w in ['Ġ'] + BYTELEVEL_BPE_RESULT],
        )

    def test_byte_level_bpe_tokenize(self):
        """
        Tests a bytelevel bpe tokenizer inside ParlAI.
        """
        parser = ParlaiParser()
        parser.set_params(
            dict_tokenizer='bytelevelbpe',
            bpe_vocab=DEFAULT_BYTELEVEL_BPE_VOCAB,
            bpe_merge=DEFAULT_BYTELEVEL_BPE_MERGE,
            bpe_add_prefix_space=False,
        )
        opt = parser.parse_args([])
        agent = DictionaryAgent(opt)
        self.assertEqual(
            # grinning face emoji
            agent.bytelevelbpe_tokenize(u'Hello, ParlAI! \U0001f600'),
            BYTELEVEL_BPE_RESULT,
        )
        self.assertEqual(
            agent.vec2txt([agent.tok2ind[w] for w in BYTELEVEL_BPE_RESULT]),
            # grinning face emoji
            u'Hello, ParlAI! \U0001f600',
        )
        self.assertEqual(
            agent.txt2vec(u'Hello, ParlAI! \U0001f600'),
            [agent.tok2ind[w] for w in BYTELEVEL_BPE_RESULT],
        )
        vocab_size = agent.bpe.tokenizer.get_vocab_size()
        with testing_utils.tempdir() as tmpdir:
            path = os.path.join(tmpdir, 'dict-checkpoint')
            agent.save(filename=path)
            agent.load(filename=path)
        # Test loading / saving
        self.assertEqual(vocab_size, agent.bpe.tokenizer.get_vocab_size())
        self.assertEqual(
            # grinning face emoji
            agent.bytelevelbpe_tokenize(u'Hello, ParlAI! \U0001f600'),
            BYTELEVEL_BPE_RESULT,
        )
        self.assertEqual(
            agent.vec2txt([agent.tok2ind[w] for w in BYTELEVEL_BPE_RESULT]),
            # grinning face emoji
            u'Hello, ParlAI! \U0001f600',
        )
        self.assertEqual(
            agent.txt2vec(u'Hello, ParlAI! \U0001f600'),
            [agent.tok2ind[w] for w in BYTELEVEL_BPE_RESULT],
        )
        # Test special token ids are mapped correctly:
        # 4 special tokens are added in ParlAI dict in the beginning and at the
        # end for Hugging Face null token would be 0 in ParlAI dict and
        # original_vocab in Hugging Face
        assert agent.txt2vec("__null__") == [0]
        assert agent.txt2vec("__start__") == [1]
        assert agent.txt2vec("__end__") == [2]
        assert agent.txt2vec("__unk__") == [3]

    def test_nofile(self):
        pp = ParlaiParser()
        DictionaryAgent.add_cmdline_args(pp, partial_opt=None)
        with self.assertRaises(IOError):
            # did not specify bpe merge or vocab
            DictionaryAgent(pp.parse_args(['--dict-tokenizer', 'bytelevelbpe']))

        with self.assertRaises(IOError):
            # specified one
            DictionaryAgent(
                pp.parse_args(
                    [
                        '--dict-tokenizer',
                        'bytelevelbpe',
                        '--bpe-merge',
                        DEFAULT_BYTELEVEL_BPE_MERGE,
                    ]
                )
            )

        with self.assertRaises(IOError):
            # specified the other
            DictionaryAgent(
                pp.parse_args(
                    [
                        '--dict-tokenizer',
                        'bytelevelbpe',
                        '--bpe-vocab',
                        DEFAULT_BYTELEVEL_BPE_VOCAB,
                    ]
                )
            )

        with self.assertRaises(IOError):
            # intentionally missing file
            DictionaryAgent(
                pp.parse_args(
                    [
                        '--dict-tokenizer',
                        'bytelevelbpe',
                        '--bpe-merge',
                        'foobar',  # intentionally wrong
                        '--bpe-vocab',
                        DEFAULT_BYTELEVEL_BPE_VOCAB,
                    ]
                )
            )

        with self.assertRaises(IOError):
            # intentionally missing file
            DictionaryAgent(
                pp.parse_args(
                    [
                        '--dict-tokenizer',
                        'bytelevelbpe',
                        '--bpe-merge',
                        DEFAULT_BYTELEVEL_BPE_MERGE,
                        '--bpe-vocab',
                        'foobar',  # intentionally wrong
                    ]
                )
            )

    def test_save_reload(self):
        """
        Save and reload an existing BL-BPE dictionary.
        """
        pp = ParlaiParser()
        DictionaryAgent.add_cmdline_args(pp, partial_opt=None)
        da = DictionaryAgent(
            pp.parse_args(
                [
                    '--dict-tokenizer',
                    'bytelevelbpe',
                    '--bpe-merge',
                    DEFAULT_BYTELEVEL_BPE_MERGE,
                    '--bpe-vocab',
                    DEFAULT_BYTELEVEL_BPE_VOCAB,
                ]
            )
        )
        # poor behavior if we failed to load
        assert da.txt2vec("hello") != []

        with testing_utils.tempdir() as tmpdir:
            newdf = os.path.join(tmpdir, "dict")
            da.save(newdf)

            # now load it
            da2 = DictionaryAgent(
                pp.parse_args(
                    ['--dict-tokenizer', 'bytelevelbpe', '--dict-file', newdf]
                )
            )
            assert da2.txt2vec("hello") == da.txt2vec("hello")

    def test_add_special_tokens(self):
        """
        Add a list of special tokens to the dictionary.
        """
        special_toks_lst = ['MY', 'NAME', 'IS', 'EMILY']
        # create Dictionary Agent
        parser = ParlaiParser()
        parser.set_params(
            dict_tokenizer='bytelevelbpe',
            bpe_vocab=DEFAULT_BYTELEVEL_BPE_VOCAB,
            bpe_merge=DEFAULT_BYTELEVEL_BPE_MERGE,
        )
        opt = parser.parse_args([])

        agent = DictionaryAgent(opt)
        agent.add_additional_special_tokens(special_toks_lst)

        self.assertEqual(agent.additional_special_tokens, special_toks_lst)
        phrases = ['Hi what is up EMILY', 'What IS your NAME', 'That is MY dog']
        for phrase in phrases:
            vec = agent.txt2vec(phrase)
            text = agent.vec2txt(vec)
            self.assertEqual(phrase, text)


class TestBuildDict(unittest.TestCase):
    def _run_test(self, opt):
        with testing_utils.tempdir() as tmpdir:
            dict_file = os.path.join(tmpdir, "dict")
            pp = build_dict.setup_args()
            pp.set_defaults(**opt)
            pp.set_defaults(task='babi')
            popt = pp.parse_args([])
            popt['dict_file'] = dict_file
            for k, v in opt.items():
                popt[k] = v

    def test_build_space(self):
        self._run_test({'dict_tokenizer': 'space'})

    def test_build_split(self):
        self._run_test({'dict_tokenizer': 'split'})

    def test_build_bpe(self):
        self._run_test({'dict_tokenizer': 'bpe', 'max_tokens': 50})


class TestGpt2HFInterop(unittest.TestCase):
    """
    Test for SlowBytelevelBPE.

    Essentially, test whether using a stand-in GPT2 tokenizer for a dict originally
    built with HF's tokenizer produces the same results.
    """

    def _get_dict_opt(self, tokenizer: str):
        parser = ParlaiParser()
        parser.set_params(
            dict_tokenizer=tokenizer,
            bpe_vocab=DEFAULT_BYTELEVEL_BPE_VOCAB,
            bpe_merge=DEFAULT_BYTELEVEL_BPE_MERGE,
            bpe_add_prefix_space=False,
            dict_loaded=True,
        )
        opt = parser.parse_args([])
        return opt

    def _run_test(self, slow_bytelevel_bpe, hf_bpe):
        """
        run the actual test.
        """
        self.assertEqual(
            # grinning face emoji
            slow_bytelevel_bpe.bytelevelbpe_tokenize(u'Hello, ParlAI! \U0001f600'),
            slow_bytelevel_bpe_RESULT,
        )
        self.assertEqual(
            slow_bytelevel_bpe.vec2txt(
                [slow_bytelevel_bpe.tok2ind[w] for w in slow_bytelevel_bpe_RESULT]
            ),
            # grinning face emoji
            u'Hello, ParlAI! \U0001f600',
        )
        self.assertEqual(
            slow_bytelevel_bpe.txt2vec(u'Hello, ParlAI! \U0001f600'),
            [slow_bytelevel_bpe.tok2ind[w] for w in slow_bytelevel_bpe_RESULT],
        )
        vocab_size = len(slow_bytelevel_bpe.bpe.encoder)
        with testing_utils.tempdir() as tmpdir:
            path = os.path.join(tmpdir, 'dict-checkpoint')
            slow_bytelevel_bpe.save(filename=path)
            slow_bytelevel_bpe.load(filename=path)
        # Test loading / saving
        self.assertEqual(vocab_size, len(slow_bytelevel_bpe.bpe.encoder))

        # next, check that hf_bpe and slow_bytelevel_bpe are equivalent
        self.assertEqual(
            slow_bytelevel_bpe.vec2txt(
                [slow_bytelevel_bpe.tok2ind[w] for w in slow_bytelevel_bpe_RESULT]
            ),
            hf_bpe.vec2txt([hf_bpe.tok2ind[w] for w in BYTELEVEL_BPE_RESULT]),
        )

    @pytest.mark.nofbcode
    def test_gpt2standin(self):
        with testing_utils.tempdir() as tmpdir:
            # we need to build the dict file
            hf_bpe_opt = self._get_dict_opt('bytelevelbpe')
            slow_bytelevel_bpe_opt = self._get_dict_opt('slow_bytelevel_bpe')

            dict_file = os.path.join(tmpdir, "dict")
            pp = build_dict.setup_args()
            pp.set_defaults(**hf_bpe_opt)
            pp.set_defaults(task='babi')
            popt = pp.parse_args([])
            popt['dict_file'] = dict_file
            build_dict.build_dict(popt)

            hf_bpe_opt['dict_file'] = dict_file
            hf_bpe = DictionaryAgent(hf_bpe_opt)

            slow_bytelevel_bpe_opt['dict_file'] = dict_file
            slow_bytelevel_bpe = DictionaryAgent(slow_bytelevel_bpe_opt)

            self._run_test(slow_bytelevel_bpe, hf_bpe)

            slow_bytelevel_bpe_opt['bpe_add_prefix_space'] = True
            slow_bytelevel_bpe = DictionaryAgent(slow_bytelevel_bpe_opt)
            self._run_prefix_space_test(slow_bytelevel_bpe)

    def _run_prefix_space_test(self, agent):
        """
        Tests gpt2standin can handle prefix space.
        """
        self.assertEqual(
            # grinning face emoji
            agent.bytelevelbpe_tokenize(u'Hello, ParlAI! \U0001f600'),
            ['\\xc4\\xa0'] + slow_bytelevel_bpe_RESULT,
        )
        self.assertEqual(
            agent.vec2txt(
                [agent.tok2ind[w] for w in ['\\xc4\\xa0'] + slow_bytelevel_bpe_RESULT]
            ),
            # grinning face emoji
            u'Hello, ParlAI! \U0001f600',
        )
        self.assertEqual(
            agent.txt2vec(u'Hello, ParlAI! \U0001f600'),
            [agent.tok2ind[w] for w in ['\\xc4\\xa0'] + slow_bytelevel_bpe_RESULT],
        )


class SpecialTokenTests(unittest.TestCase):
    """
    Test special tokens tokenization.
    """

    def _run_specialtok_test(self, **kwargs):
        for special_token in ['SPECIAL TOKENS', '[SPECIAL; TOKENS]']:
            with testing_utils.tempdir() as tmpdir:
                if 'dict_file' not in kwargs:
                    kwargs['dict_file'] = os.path.join(tmpdir, 'dict')
                string = f"This is a test of {special_token}"
                parser = ParlaiParser(False, False)
                DictionaryAgent.add_cmdline_args(parser, partial_opt=None)
                opt = parser.parse_kwargs(**kwargs)
                da = DictionaryAgent(opt)
                before = da.tokenize(string)
                da.add_additional_special_tokens([special_token])
                after = da.tokenize(string)
                assert before != after
                assert len(before) > len(after)
                assert after[-1] == special_token
                assert before[:5] == after[:5]
                if opt['dict_tokenizer'] in (
                    'bytelevelbpe',
                    'gpt2',
                    'slow_bytelevel_bpe',
                ):
                    # we need to let the dictionary handle the tokenid mappings
                    assert da.vec2txt(da.txt2vec(string)) == string

    def test_specialtok_slow_bytelevel_bpe(self):
        self._run_specialtok_test(
            dict_tokenizer="slow_bytelevel_bpe", dict_file="zoo:blender/dict_3B/dict"
        )

    @unittest.skipUnless(TOKENIZERS, "No tokenizers available")
    def test_specialtok_bytelevelbpe(self):
        self._run_specialtok_test(
            dict_tokenizer="bytelevelbpe", dict_file="zoo:blender/dict_3B/dict"
        )

    def test_specialtok_gpt2(self):
        self._run_specialtok_test(dict_tokenizer="gpt2")

    def test_specialtok_re(self):
        self._run_specialtok_test(dict_tokenizer='re')

    def test_specialtok_space(self):
        self._run_specialtok_test(dict_tokenizer='space')

    def test_specialtok_split(self):
        self._run_specialtok_test(dict_tokenizer='split')

    def test_specialtok_nonsupport(self):
        for tokenizer in ["bpe"]:
            with self.assertRaises(NotImplementedError):
                self._run_specialtok_test(dict_tokenizer=tokenizer)


class TestBpeDropout(unittest.TestCase):
    def _test_bpe_dropout(self, **dict_args):
        pp = ParlaiParser(False, False)
        DictionaryAgent.add_cmdline_args(pp, partial_opt=None)
        opt = pp.parse_kwargs(bpe_dropout=0.5, **dict_args)
        da = DictionaryAgent(opt)
        da.set_tokenization_mode(TokenizationMode.TEST_TIME_TEXT)
        s = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Donec vitae metus sollicitudin, ullamcorper tortor ut, rhoncus lacus. "
            "Praesent sollicitudin commodo turpis, ut pharetra tortor gravida nec."
        )
        no_dropout = da.txt2vec(s)
        da.set_tokenization_mode(TokenizationMode.TRAIN_TIME_TEXT)
        not_the_same = 0
        for _ in range(30):
            r = da.txt2vec(s)
            assert da.vec2txt(r) == s
            if r != no_dropout:
                not_the_same += 1
        assert not_the_same > 0

    def test_gpt2_bpe_dropout(self):
        self._test_bpe_dropout(dict_tokenizer='gpt2')

    def test_slowbytelevel_dropout(self):
        self._test_bpe_dropout(
            dict_tokenizer="slow_bytelevel_bpe", dict_file="zoo:blender/dict_3B/dict"
        )

    def test_bytelevelbpe_dropout(self):
        with self.assertRaises(NotImplementedError):
            self._test_bpe_dropout(
                dict_tokenizer="bytelevelbpe", dict_file="zoo:blender/dict_3B/dict"
            )
