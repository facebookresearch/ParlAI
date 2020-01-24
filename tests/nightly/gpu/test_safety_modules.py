#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.utils.safety import OffensiveLanguageClassifier
from parlai.utils.safety import OffensiveStringMatcher
from parlai.utils.testing import skipUnlessBPE, skipUnlessGPU


import unittest

DEFINITELY_GOOD = [
    'hey how are you',
    'what is your name?',
    'i like to play sports',
    'i have a horse',
]
PROBABLY_BAD = ['i hate you', 'you are pretty ugly', 'parlai is the worst platform']
DEFINITELY_BAD = ['fuck you', 'you are a piece of shit']


class TestSafetyModules(unittest.TestCase):
    def test_string_matcher(self):
        sm = OffensiveStringMatcher()
        for phrase in DEFINITELY_BAD:
            assert phrase in sm, f'`{phrase}` is offensive'
        for phrase in DEFINITELY_GOOD:
            assert phrase not in sm, f'`{phrase}` is not offensive'

    @skipUnlessGPU
    @skipUnlessBPE
    def test_classifier(self):
        lc = OffensiveLanguageClassifier()
        for phrase in DEFINITELY_BAD:
            assert phrase in lc, f'`{phrase}` is offensive'
        for phrase in PROBABLY_BAD:
            assert phrase in lc, f'`{phrase}` is offensive'
        for phrase in DEFINITELY_GOOD:
            assert phrase not in lc, f'`{phrase}` is not offensive'


if __name__ == '__main__':
    unittest.main()
