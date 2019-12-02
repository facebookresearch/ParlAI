# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import os.path
import unittest
import sys

from aiml import Kernel



class TestEncoding( unittest.TestCase ):

    longMessage = True

    def setUp(self):
        self.k = Kernel()
        self.k.bootstrap( learnFiles='encoding.aiml', 
                          chdir=os.path.dirname(__file__) )

    def tearDown(self):
        del self.k

    def _testTag(self, input_, outputList, name=None, encoding='utf-8'):
        """Test by feeding the Kernel 'input'.  If the result
        matches any of the strings in 'outputList', the test passes.
        """
        
        print( b"Testing <" + (name or input_).encode(encoding) + b">")
        response = self.k._cod.dec( self.k.respond( self.k._cod.enc(input_) ) )
        self.assertIn( response, outputList, msg="input=%s"%input_ )

    def test01_utf8( self ):
        '''Test literal pattern'''
        self._testTag( u'pattern with eñe', [u"pattern #1 matched!"])

    def test02_utf8( self ):
        '''Test star pattern'''
        self._testTag( u'pattern with Á', [u"pattern #2 matched: Á"])

    def test03_noencoding( self ):
        '''Test unencoded strings'''
        self.k.setTextEncoding( False )
        self._testTag( u'pattern with eñe', [u"pattern #1 matched!"])
        self._testTag( u'pattern with Á', [u"pattern #2 matched: Á"])

    def test04_iso8859( self ):
        enc = 'iso-8859-1'
        self.k.setTextEncoding( enc )
        self._testTag( u'pattern with Á', [u"pattern #2 matched: Á"],
                       encoding=enc)
