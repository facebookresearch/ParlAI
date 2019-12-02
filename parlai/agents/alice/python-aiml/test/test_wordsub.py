# -*- coding: latin-1 -*-

from __future__ import print_function
import unittest

from aiml.WordSub import WordSub


class TestWordSub( unittest.TestCase ):

    longMessage = True

    def setUp(self):
        self.subber = WordSub()
        self.subber["apple"] = "banana"
        self.subber["orange"] = "pear"
        self.subber["banana" ] = "apple"
        self.subber["he"] = "she"
        self.subber["I'd"] = "I would"

    def tearDown(self):
        del self.subber

    def test01_sub( self ):
        '''test wordsub'''
        inStr = "He said he'd like to go with me"
        outStr = "She said she'd like to go with me"
        self.assertEqual( outStr, self.subber.sub(inStr) )

    def test02_case( self ):
        '''test case insensitivity'''
        inStr =  "I'd like one apple, one Orange and one BANANA."
        outStr = "I Would like one banana, one Pear and one APPLE."
        self.assertEqual( outStr, self.subber.sub(inStr) )

