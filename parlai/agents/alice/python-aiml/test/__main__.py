# A script to enable execution of any python unit test in this directory.
# It will be called when Python tries to 'execute' this directory
"""
Usage:

  * execute all unit tests in this dir
       python <dir> [options] [-a]
 
  * execute a certain set of tests
       python <dir> [options] <name1> <name2> ... 

Options:
  -f:  failfast (abort with the first test that fails)
  -v, -vv: increase verbosity
  -q: reduce verbosity
  -d LEVEL: set debug level
"""

from __future__ import print_function

import re
import unittest
import os.path
import sys


# --------------------------------------------------------------------

def script_list( path ):
    """
    Get all the python scripts in a directory. Scripts are taken as all 
    files ending in ``.py`` that do not start with an underscore or dot.
    """
    regexp = re.compile( r'^([^_\.].+)\.py$' )
    r = []
    for f in os.listdir(path):
        if os.path.isdir( os.path.join(path,f) ):
            continue
        m = regexp.match(f)
        if m:
            r.append( m.group(1) )
    return sorted(r)


# --------------------------------------------------------------------

def default_opts():
    '''Default options'''
    return type( "options", (), 
                { 'help' : False,
                  'all' : False,
                  'quiet' : False,
                  'failfast': False,
                  'verbosity': 1 } ) 

def read_opts():
    '''Read command-line options'''
    # Default options
    opt = default_opts()
    # Detect options
    while len(sys.argv) > 1 and sys.argv[1].startswith('-'):
        if sys.argv[1] == '-h':
            opt.help = True
        elif sys.argv[1] == '-q':
            opt.quiet = True
        elif sys.argv[1] == '-v':
            opt.verbosity = 2
        elif sys.argv[1] == '-vv':
            opt.verbosity = 3
        elif sys.argv[1] in ('-a','-all','--all'):
            opt.all = True
        elif sys.argv[1] in ('-f'):
            opt.failfast = True
        elif sys.argv[1] == '-d':
            import logging
            logging.basicConfig( level=sys.argv[2], 
                                 format='%(levelname)s %(filename)s %(funcName)s: %(message)s' )
            sys.argv.pop(1)
        sys.argv.pop(1)    # remove the just-used argument
    return opt


# --------------------------------------------------------------------

def load_tests( opt=None ):
    '''Load the test suite by adding all python files in this directory'''

    if opt is None:
        opt = default_opts()
    thisdir = os.path.dirname(__file__)

    # Select the test(s) to run
    if __name__ != '__main__' or len(sys.argv) < 2 or opt.all:
        test_list = script_list( thisdir )
    else:
        test_list = [ t if t.startswith('test_') else 'test_' + t
                      for t in sys.argv[1:] ]

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()

    # Add the base path of python-aiml sources
    sys.path.insert( 0, os.path.dirname(thisdir) )

    # Add all requested unit tests
    if not opt.quiet:
        print( "\n******************************\n" )
    for test in test_list:
        if not opt.quiet:
            print( "Loading unit test '" + test + "'" )
        suite.addTest( loader.loadTestsFromName( 'test.' + test ) )

    return suite


if __name__ == '__main__':
    # Usage message
    opt = read_opts()
    if opt.help:
        print( __doc__ )
        sys.exit(1)
    # Load the tests
    suite = load_tests( opt )
    # Run the tests
    if not opt.quiet:
        print( "\n\nRunning test suite\n" )
    try:
        testrunner = unittest.TextTestRunner( verbosity=opt.verbosity )
        testrunner.failfast = opt.failfast
        testrunner.run( suite )
    except KeyboardInterrupt as err:
        print( "Interrupted!! :", err )
        sys.exit(1)
