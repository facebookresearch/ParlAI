#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import find_ngrams
from parlai.core.params import ParlaiParser
import parlai.core.testing_utils as testing_utils
import os
import shutil
import unittest


class TestDictionary(unittest.TestCase):
    """Basic tests on the built-in parlai Dictionary."""

    def test_find_ngrams(self):
        """Can the ngram class properly recognize uni, bi, and trigrams?"""
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
        """Check that the dictionary is correctly adding and parsing short
        sentence.
        """
        from parlai.core.dict import DictionaryAgent
        from parlai.core.params import ParlaiParser

        argparser = ParlaiParser()
        DictionaryAgent.add_cmdline_args(argparser)
        opt = argparser.parse_args(print_args=False)
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
        """Check that moving a model without moving the dictionary raises the
        appropriate error.
        """
        # Download model, move to a new location
        datapath = ParlaiParser().parse_args(print_args=False)['datapath']
        try:
            # remove unittest models if there before
            shutil.rmtree(os.path.join(datapath, 'models/unittest'))
        except FileNotFoundError:
            pass
        testing_utils.download_unittest_models()

        zoo_path = 'models:unittest/seq2seq/model'
        model_path = modelzoo_path(datapath, zoo_path)
        os.remove(model_path + '.dict')
        # Test that eval model fails
        with self.assertRaises(RuntimeError):
            testing_utils.eval_model(dict(
                task='babi:task1k:1',
                model_file=model_path
            ))
        try:
            # remove unittest models if there after
            shutil.rmtree(os.path.join(datapath, 'models/unittest'))
        except FileNotFoundError:
            pass

    def test_train_model_with_no_dict_file(self):
        """Check that attempting to train a model without specifying a dict_file
        or model_file fails
        """
        import parlai.scripts.train_model as tms
        with testing_utils.capture_output():
            parser = tms.setup_args()
            parser.set_params(
                task='babi:task1k:1',
                model='seq2seq'
            )
            popt = parser.parse_args(print_args=False)
            with self.assertRaises(RuntimeError):
                tms.TrainLoop(popt)


if __name__ == '__main__':
    unittest.main()
