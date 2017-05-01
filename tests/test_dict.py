# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.dict import find_ngrams
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
        opt = argparser.parse_args()
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


if __name__ == '__main__':
    unittest.main()
