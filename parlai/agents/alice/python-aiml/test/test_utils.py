# -*- coding: latin-1 -*-

from __future__ import print_function
import time
import os.path
import unittest

from aiml import Utils


class TestUtils( unittest.TestCase ):

    longMessage = True

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sentences( self ):
        sents = Utils.sentences("First.  Second, still?  Third and Final!  Well, not really")
        self.assertEqual( 4, len(sents) )

