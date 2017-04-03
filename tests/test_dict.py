#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.core.dict import find_ngrams
import unittest


class TestDictionary(unittest.TestCase):

    def test_find_ngrams(self):
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


if __name__ == '__main__':
    unittest.main()
