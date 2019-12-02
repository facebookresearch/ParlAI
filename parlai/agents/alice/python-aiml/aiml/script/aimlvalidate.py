"""
Python AIML Validator, v1.1
Author: Cort Stratton (cort@cortstratton.org)

Usage:
    aimlvalidate.py file1.aiml [file2.aiml ...]
"""

# Revision history:
#
# 1.0.1: Redirected stderr to stdout
# 1.0: Initial release

from __future__ import print_function

import sys
import io
import glob
import xml.sax

import aiml

PY3 = sys.version[0] == 3


def get_file_position( filename, row, col, encoding='utf-8' ):
    '''
    Find a place within a file
    '''
    # Get the line
    with io.open( filename, 'rb' ) as file:
        lines = file.readlines()
        start = col-25 if col>25 else 0
        buf = lines[row-1][start:start+50]
    # Decode it
    try:
        buf = buf.decode(encoding,'replace')
    except UnicodeDecodeError:
        pass
    buf = buf.rstrip()
    # Get a marker pointing to the column
    marker = '-' * (col-start) + '^'
    if start > 0:
        buf = u'...' + buf
        marker = '---' + marker
    if start+50 < len(lines[row-1]):
        buf += u'...'
                #buf = buf + b'\n' + marker
                #LOG.warn( u'%s :\n%s', e,  buf )
    return buf, marker


def main():
    '''Entry point'''
    # Need input file(s)!
    if len(sys.argv) < 2:
        print( __doc__ )
        sys.exit(2)

    # AimlParser prints its errors to stderr; we redirect stderr to stdout.
    sys.stderr = sys.stdout

    # Iterate over input files    
    validCount = 0
    docCount = 0
    for arg in sys.argv[1:]:
        # Input files can contain wildcards; iterate over matches
        for f in glob.glob(arg):
            parser = xml.sax.make_parser(["aiml.AimlParser"])
            handler = parser.getContentHandler()
            docCount += 1
            print( "Validating %s:" % f, end=' ' )
            try:
                #with open(f,'rb') as inp:
                #    src = xml.sax.xmlreader.InputSource()
                #    src.setByteStream(f)
                #    #input_source.setEncoding(encoding)
                #    parser.parse(src)
                ## Attempt to parse the file.
                parser.parse(f)
                # Check the number of parse errors.
                if handler.getNumErrors() == 0:    
                    validCount += 1
                    print( "PASSED\n" )
                else:
                    print( "FAILED\n" )
            except xml.sax.SAXParseException as err:
                # These errors occur if the document does not contain
                # well-formed XML (e.g. open or unbalanced tags).  If
                # they occur, parsing the whole document is aborted
                # immediately.
                row, col = err.getLineNumber(), err.getColumnNumber()
                # Find where the parser broke
                errbuf, below = get_file_position( f, row, col )
                # Prepare an error message
                msg = u'{}: row={} col={} id={}:\n{}\n{}'.format(
                    err.getMessage(), row, col, err.getSystemId(), 
                    errbuf, below )
                print( "\n  FATAL ERROR: %s\n" % msg )
                
    # Print final results
    print( "%d out of %d documents are AIML 1.0.1 compliant." % (validCount, docCount))
    if docCount == validCount:
        print( "Congratulations!" )
    else:
        print( """For help resolving syntax errors, refer to the AIML 1.0.1 specification
available on the web at: http://alicebot.org/TR/2001/WD-aiml""")


if __name__ == "__main__":
    main()
