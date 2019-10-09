#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.safety import OffensiveLanguageClassifier as olc
from parlai.utils.safety import OffensiveStringMatcher as osm


import unittest

definitely_good = [
    'hey how are you',
    'what is your name?',
    'i like to play sports',
    'i have a horse',
]
probably_bad = ['i hate you', 'you are pretty ugly', 'parlai is the worst platform']
definitely_bad = ['fuck you', 'you are a piece of shit']


class TestSafetyModules(unittest.TestCase):
    def test_string_matcher(self):
        sm = osm()
        for phrase in definitely_bad:
            assert phrase in sm, f'`{phrase}` is offensive'
        for phrase in definitely_good:
            assert phrase not in sm, f'`{phrase}` is not offensive'

    def test_classifier(self):
        lc = olc()
        for phrase in definitely_bad:
            assert phrase in lc, f'`{phrase}` is offensive'
        for phrase in probably_bad:
            assert phrase in lc, f'`{phrase}` is offensive'
        for phrase in definitely_good:
            assert phrase not in lc, f'`{phrase}` is not offensive'


if __name__ == '__main__':
    unittest.main()
