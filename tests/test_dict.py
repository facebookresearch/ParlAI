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
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt

import parlai.utils.testing as testing_utils
import os
import shutil
import unittest


class TestDictionary(unittest.TestCase):
    """
    Basic tests on the built-in parlai Dictionary.
    """

    def test_gpt2_bpe_tokenize(self):
        opt = Opt({'dict_tokenizer': 'gpt2', 'datapath': './data'})
        agent = DictionaryAgent(opt)
        self.assertEqual(
            # grinning face emoji
            agent.gpt2_tokenize(u'Hello, ParlAI! \U0001f600'),
            [
                'Hello',
                ',',
                r'\xc4\xa0Par',
                'l',
                'AI',
                '!',
                r'\xc4\xa0\xc3\xb0\xc5\x81\xc4\xba',
                r'\xc4\xa2',
            ],
        )
        self.assertEqual(
            agent.vec2txt(
                agent.tok2ind[w]
                for w in [
                    'Hello',
                    ',',
                    r'\xc4\xa0Par',
                    'l',
                    'AI',
                    '!',
                    r'\xc4\xa0\xc3\xb0\xc5\x81\xc4\xba',
                    r'\xc4\xa2',
                ]
            ),
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
        argparser = ParlaiParser()
        DictionaryAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args([], print_args=False)
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

    def test_set_model_file_without_dict_file(self):
        """
        Check that moving a model without moving the dictfile raises an error.
        """
        # Download model, move to a new location
        datapath = ParlaiParser().parse_args([], print_args=False)['datapath']
        try:
            # remove unittest models if there before
            shutil.rmtree(os.path.join(datapath, 'models/unittest'))
        except FileNotFoundError:
            pass

        zoo_path = 'zoo:unittest/seq2seq/model'
        model_path = modelzoo_path(datapath, zoo_path)
        os.remove(model_path + '.dict')
        # Test that eval model fails
        with self.assertRaises(RuntimeError):
            testing_utils.eval_model(dict(task='babi:task1k:1', model_file=model_path))
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
        popt = parser.parse_args([], print_args=False)
        with self.assertRaises(RuntimeError):
            tms.TrainLoop(popt)


if __name__ == '__main__':
    unittest.main()
