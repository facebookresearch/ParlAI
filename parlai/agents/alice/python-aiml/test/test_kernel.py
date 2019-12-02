# -*- coding: latin-1 -*-

from __future__ import print_function
import time
import os.path
import unittest

from aiml import Kernel



class TestKernel( unittest.TestCase ):

    longMessage = True

    def setUp(self):
        self.k = Kernel()
        testfile = os.path.join(os.path.dirname(__file__),"self-test.aiml")
        self.k.bootstrap(learnFiles=testfile)

    def tearDown(self):
        del self.k

    def _testTag(self, tag, input_, outputList):
        """Tests 'tag' by feeding the Kernel 'input'.  If the result
        matches any of the strings in 'outputList', the test passes.

        """
        print( "Testing <" + tag + ">", end='\n' )
        # Send the input and collect the output
        response = self.k._cod.dec( self.k.respond( self.k._cod.enc(input_) ) )
        # Check that output is as expected
        self.assertIn( response, outputList, msg="input=%s"%input_ )

    def test01_bot( self ):
        self._testTag('bot', 'test bot', ["My name is Nameless"])

    def test02_condition( self ):
        self.k.setPredicate('gender', 'male')
        self._testTag('condition test #1', 'test condition name value', ['You are handsome'])
        self.k.setPredicate('gender', 'female')
        self._testTag('condition test #2', 'test condition name value', [''])
        self._testTag('condition test #3', 'test condition name', ['You are beautiful'])
        self.k.setPredicate('gender', 'robot')
        self._testTag('condition test #4', 'test condition name', ['You are genderless'])
        self._testTag('condition test #5', 'test condition', ['You are genderless'])
        self.k.setPredicate('gender', 'male')
        self._testTag('condition test #6', 'test condition', ['You are handsome'])

    def test03_date( self ):
        # the date test will occasionally fail if the original and "test"
        # times cross a second boundary.  There's no good way to avoid
        # this problem and still do a meaningful test, so we simply
        # provide a friendly message to be printed if the test fails.
        date_warning = """
        NOTE: the <date> test will occasionally report failure even if it
        succeeds.  So long as the response looks like a date/time string,
        there's nothing to worry about.
        """
        if not self._testTag('date', 'test date', ["The date is %s" % time.asctime()]):
            print( date_warning )

    def test04_case( self ):
        self._testTag('formal', 'test formal', ["Formal Test Passed"])
        self._testTag('lowercase', 'test lowercase', ["The Last Word Should Be lowercase"])
        self._testTag('sentence', "test sentence", ["My first letter should be capitalized."])
        self._testTag('uppercase', 'test uppercase', ["The Last Word Should Be UPPERCASE"])

    def test05_gender( self ):
        self._testTag('gender', 'test gender', ["He'd told her he heard that her hernia is history"])

    def test06_getset( self ):
        self._testTag('get/set', 'test get and set', ["I like cheese. My favorite food is cheese"])

    def test07_notimplemented( self ):
        self._testTag('gossip', 'test gossip', ["Gossip is not yet implemented"])
        self._testTag('javascript', 'test javascript', ["Javascript is not yet implemented"])

    def test08_id( self ):
        self._testTag('id', 'test id', ["Your id is _global"])

    def test09_input( self ):
        self._testTag('input', 'test input', ['You just said: test input'])


    def test10_person( self ):
        self._testTag('person', 'test person', ['HE think i knows that my actions threaten him and his.'])
        self._testTag('person2', 'test person2', ['YOU think me know that my actions threaten you and yours.'])
        self._testTag('person2 (no contents)', 'test person2 I Love Lucy', ['YOU Love Lucy'])


    def test11_random( self ):
        self._testTag('random', 'test random', ["response #1", "response #2", "response #3"])
        self._testTag('random empty', 'test random empty', ["Nothing here!"])

    def test12_size( self ):
        self._testTag('size', "test size", ["I've learned %d categories" % self.k.numCategories()])

    def test13_srai( self ):
        self._testTag('sr', "test sr test srai", ["srai results: srai test passed"])
        self._testTag('sr nested', "test nested sr test srai", ["srai results: srai test passed"])
        self._testTag('srai', "test srai", ["srai test passed"])
        self._testTag('srai infinite', "test srai infinite", [""])

    def test13_star( self ):
        self._testTag('star test #1', 'You should test star begin', ['Begin star matched: You should']) 
        self._testTag('star test #2', 'test star creamy goodness middle', ['Middle star matched: creamy goodness'])
        self._testTag('star test #3', 'test star end the credits roll', ['End star matched: the credits roll'])
        self._testTag('star test #4', 'test star having multiple stars in a pattern makes me extremely happy',
                 ['Multiple stars matched: having, stars in a pattern, extremely happy'])

    def test14_that( self ):
        self._testTag('system', "test system", ["The system says hello!"])
        # This one must go right after the previous one
        self._testTag('that test #1', "test that", ["I just said: The system says hello!"])
        self._testTag('that test #2', "test that", ["I have already answered this question"])

    def test15_thatstar( self ):
        self._testTag('thatstar test #1', "test thatstar", ["I say beans"])
        self._testTag('thatstar test #2', "test thatstar", ["I just said \"beans\""])
        self._testTag('thatstar test #3', "test thatstar multiple", ['I say beans and franks for everybody'])
        self._testTag('thatstar test #4', "test thatstar multiple", ['Yes, beans and franks for all!'])

    def test15_think( self ):
        self._testTag('think', "test think", [""])

    def test16_topic( self ):
        self.k.setPredicate("topic", "fruit")
        self._testTag('topic', "test topic", ["We were discussing apples and oranges"]) 
        self.k.setPredicate("topic", "Soylent Green")
        self._testTag('topicstar test #1', 'test topicstar', ["Solyent Green is made of people!"])
        self.k.setPredicate("topic", "Soylent Ham and Cheese")
        self._testTag('topicstar test #2', 'test topicstar multiple', ["Both Soylents Ham and Cheese are made of people!"])

    def test17_unicode( self ):
        self._testTag('unicode support', u"ÔÇÉÏºÃ", [u"Hey, you speak Chinese! ÔÇÉÏºÃ"])

    def test18_version( self ):
        self._testTag('version', 'test version', ["PyAIML is version %s" % self.k.version()])

    def test18_whitespace( self ):
        self._testTag('whitespace preservation', 'test whitespace', ["Extra   Spaces\n   Rule!   (but not in here!)    But   Here   They   Do!"])

        # Run an interactive interpreter
        #print( "\nEntering interactive mode (ctrl-c to exit)" )
        #while True: print( self.k.respond(raw_input("> ")) )
