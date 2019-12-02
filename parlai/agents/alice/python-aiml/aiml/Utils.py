"""This file contains assorted general utility functions used by other
modules in the PyAIML package.

"""

def sentences(s):
    """Split the string s into a list of sentences."""
    try: s+""
    except: raise TypeError( "s must be a string" )
    pos = 0
    sentenceList = []
    l = len(s)
    while pos < l:
        try: p = s.index('.', pos)
        except: p = l+1
        try: q = s.index('?', pos)
        except: q = l+1
        try: e = s.index('!', pos)
        except: e = l+1
        end = min(p,q,e)
        sentenceList.append( s[pos:end].strip() )
        pos = end+1
    # If no sentences were found, return a one-item list containing
    # the entire input string.
    if len(sentenceList) == 0: sentenceList.append(s)
    return sentenceList

