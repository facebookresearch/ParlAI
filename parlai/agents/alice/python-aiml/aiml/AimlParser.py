'''
A parser for AIML files
'''

from __future__ import print_function

from xml.sax.handler import ContentHandler
from xml.sax.xmlreader import Locator
import sys
import xml.sax
import xml.sax.handler

from .constants import *


class AimlParserError(Exception): 
    pass


class AimlHandler(ContentHandler):
    '''
    A SAX handler for AIML files
    '''

    # The legal states of the AIML parser
    _STATE_OutsideAiml    = 0
    _STATE_InsideAiml     = 1
    _STATE_InsideCategory = 2
    _STATE_InsidePattern  = 3
    _STATE_AfterPattern   = 4
    _STATE_InsideThat     = 5
    _STATE_AfterThat      = 6
    _STATE_InsideTemplate = 7
    _STATE_AfterTemplate  = 8


    def __init__(self, encoding=None):
        self.categories = {}
        self._encoding = encoding
        self._state = self._STATE_OutsideAiml
        self._version = ""
        self._namespace = ""
        self._forwardCompatibleMode = False
        self._currentPattern = ""
        self._currentThat    = ""
        self._currentTopic   = ""
        self._insideTopic = False
        self._currentUnknown = "" # the name of the current unknown element

        # This is set to true when a parse error occurs in a category.
        self._skipCurrentCategory = False

        # Counts the number of parse errors in a particular AIML document.
        # query with getNumErrors().  If 0, the document is AIML-compliant.
        self._numParseErrors = 0

        # TODO: select the proper validInfo table based on the version number.
        self._validInfo = self._validationInfo101

        # This stack of bools is used when parsing <li> elements inside
        # <condition> elements, to keep track of whether or not an
        # attribute-less "default" <li> element has been found yet.  Only
        # one default <li> is allowed in each <condition> element.  We need
        # a stack in order to correctly handle nested <condition> tags.
        self._foundDefaultLiStack = []

        # This stack of strings indicates what the current whitespace-handling
        # behavior should be.  Each string in the stack is either "default" or
        # "preserve".  When a new AIML element is encountered, a new string is
        # pushed onto the stack, based on the value of the element's "xml:space"
        # attribute (if absent, the top of the stack is pushed again).  When
        # ending an element, pop an object off the stack.
        self._whitespaceBehaviorStack = ["default"]
        
        self._elemStack = []
        self._locator = Locator()
        self.setDocumentLocator(self._locator)

    def getNumErrors(self):
        "Return the number of errors found while parsing the current document."
        return self._numParseErrors

    def setEncoding(self, encoding):
        """
        Set the text encoding to use when encoding strings read from XML.
        Defaults to no encoding.
        """
        self._encoding = encoding

    def _location(self):
        "Return a string describing the current location in the source file."
        line = self._locator.getLineNumber()
        column = self._locator.getColumnNumber()
        return "(line %d, column %d)" % (line, column)

    def _pushWhitespaceBehavior(self, attr):
        """Push a new string onto the whitespaceBehaviorStack.

        The string's value is taken from the "xml:space" attribute, if it exists
        and has a legal value ("default" or "preserve").  Otherwise, the previous
        stack element is duplicated.
        """
        assert len(self._whitespaceBehaviorStack) > 0, "Whitespace behavior stack should never be empty!"
        try:
            if attr["xml:space"] == "default" or attr["xml:space"] == "preserve":
                self._whitespaceBehaviorStack.append(attr["xml:space"])
            else:
                raise AimlParserError( "Invalid value for xml:space attribute "+self._location() )
        except KeyError:
            self._whitespaceBehaviorStack.append(self._whitespaceBehaviorStack[-1])

    def startElementNS(self, name, qname, attr):
        print( "QNAME:", qname )
        print( "NAME:", name )
        uri,elem = name
        if (elem == "bot"): 
                    print( "name:", attr.getValueByQName("name"), "a'ite?" )
        self.startElement(elem, attr)
        pass

    def startElement(self, name, attr):
        # Wrapper around _startElement, which catches errors in _startElement()
        # and keeps going.
        
        # If we're inside an unknown element, ignore everything until we're
        # out again.
        if self._currentUnknown != "":
            return
        # If we're skipping the current category, ignore everything until
        # it's finished.
        if self._skipCurrentCategory:
            return

        # process this start-element.
        try: self._startElement(name, attr)
        except AimlParserError as err:
            # Print the error message
            sys.stderr.write("PARSE ERROR: %s\n" % err)
            
            self._numParseErrors += 1 # increment error count
            # In case of a parse error, if we're inside a category, skip it.
            if self._state >= self._STATE_InsideCategory:
                self._skipCurrentCategory = True
            
    def _startElement(self, name, attr):
        if name == "aiml":
            # <aiml> tags are only legal in the OutsideAiml state
            if self._state != self._STATE_OutsideAiml:
                raise AimlParserError( "Unexpected <aiml> tag "+self._location() )
            self._state = self._STATE_InsideAiml
            self._insideTopic = False
            self._currentTopic = u""
            try: self._version = attr["version"]
            except KeyError:
                # This SHOULD be a syntax error, but so many AIML sets out there are missing
                # "version" attributes that it just seems nicer to let it slide.
                #raise AimlParserError( "Missing 'version' attribute in <aiml> tag "+self._location() )
                #print( "WARNING: Missing 'version' attribute in <aiml> tag "+self._location() )
                #print( "         Defaulting to version 1.0" )
                self._version = "1.0"
            self._forwardCompatibleMode = (self._version != "1.0.1")
            self._pushWhitespaceBehavior(attr)          
            # Not sure about this namespace business yet...
            #try:
            #   self._namespace = attr["xmlns"]
            #   if self._version == "1.0.1" and self._namespace != "http://alicebot.org/2001/AIML-1.0.1":
            #       raise AimlParserError( "Incorrect namespace for AIML v1.0.1 "+self._location() )
            #except KeyError:
            #   if self._version != "1.0":
            #       raise AimlParserError( "Missing 'version' attribute(s) in <aiml> tag "+self._location() )
        elif self._state == self._STATE_OutsideAiml:
            # If we're outside of an AIML element, we ignore all tags.
            return
        elif name == "topic":
            # <topic> tags are only legal in the InsideAiml state, and only
            # if we're not already inside a topic.
            if (self._state != self._STATE_InsideAiml) or self._insideTopic:
                raise AimlParserError( "Unexpected <topic> tag", self._location() )
            try: self._currentTopic = unicode(attr['name'])
            except KeyError:
                raise AimlParserError( "Required \"name\" attribute missing in <topic> element "+self._location() )
            self._insideTopic = True
        elif name == "category":
            # <category> tags are only legal in the InsideAiml state
            if self._state != self._STATE_InsideAiml:
                raise AimlParserError( "Unexpected <category> tag "+self._location() )
            self._state = self._STATE_InsideCategory
            self._currentPattern = u""
            self._currentThat = u""
            # If we're not inside a topic, the topic is implicitly set to *
            if not self._insideTopic: self._currentTopic = u"*"
            self._elemStack = []
            self._pushWhitespaceBehavior(attr)
        elif name == "pattern":
            # <pattern> tags are only legal in the InsideCategory state
            if self._state != self._STATE_InsideCategory:
                raise AimlParserError( "Unexpected <pattern> tag "+self._location() )
            self._state = self._STATE_InsidePattern
        elif name == "that" and self._state == self._STATE_AfterPattern:
            # <that> are legal either inside a <template> element, or
            # inside a <category> element, between the <pattern> and the
            # <template> elements.  This clause handles the latter case.
            self._state = self._STATE_InsideThat
        elif name == "template":
            # <template> tags are only legal in the AfterPattern and AfterThat
            # states
            if self._state not in [self._STATE_AfterPattern, self._STATE_AfterThat]:
                raise AimlParserError( "Unexpected <template> tag "+self._location() )
            # if no <that> element was specified, it is implicitly set to *
            if self._state == self._STATE_AfterPattern:
                self._currentThat = u"*"
            self._state = self._STATE_InsideTemplate
            self._elemStack.append(['template',{}])
            self._pushWhitespaceBehavior(attr)
        elif self._state == self._STATE_InsidePattern:
            # Certain tags are allowed inside <pattern> elements.
            if name == "bot" and "name" in attr and attr["name"] == u"name":
                # Insert a special character string that the PatternMgr will
                # replace with the bot's name.
                self._currentPattern += u" BOT_NAME "
            else:
                raise AimlParserError( ( "Unexpected <%s> tag " % name)+self._location() )
        elif self._state == self._STATE_InsideThat:
            # Certain tags are allowed inside <that> elements.
            if name == "bot" and "name" in attr and attr["name"] == u"name":
                # Insert a special character string that the PatternMgr will
                # replace with the bot's name.
                self._currentThat += u" BOT_NAME "
            else:
                raise AimlParserError( ("Unexpected <%s> tag " % name)+self._location() )
        elif self._state == self._STATE_InsideTemplate and name in self._validInfo:
            # Starting a new element inside the current pattern. First
            # we need to convert 'attr' into a native Python dictionary,
            # so it can later be marshaled.
            it = ( (unicode(k),unicode(v)) for k,v in attr.items() )
            attrDict = dict( it )
            self._validateElemStart(name, attrDict, self._version)
            # Push the current element onto the element stack.
            self._elemStack.append( [unicode(name),attrDict] )
            self._pushWhitespaceBehavior(attr)
            # If this is a condition element, push a new entry onto the
            # foundDefaultLiStack
            if name == "condition":
                self._foundDefaultLiStack.append(False)
        else:
            # we're now inside an unknown element.
            if self._forwardCompatibleMode:
                # In Forward Compatibility Mode, we ignore the element and its
                # contents.
                self._currentUnknown = name
            else:
                # Otherwise, unknown elements are grounds for error!
                raise AimlParserError( ("Unexpected <%s> tag " % name)+self._location() )

    def characters(self, ch):
        # Wrapper around _characters which catches errors in _characters()
        # and keeps going.
        if self._state == self._STATE_OutsideAiml:
            # If we're outside of an AIML element, we ignore all text
            return
        if self._currentUnknown != "":
            # If we're inside an unknown element, ignore all text
            return
        if self._skipCurrentCategory:
            # If we're skipping the current category, ignore all text.
            return
        try: self._characters(ch)
        except AimlParserError as msg:
            # Print the message
            sys.stderr.write("PARSE ERROR: %s\n" % msg)
            self._numParseErrors += 1 # increment error count
            # In case of a parse error, if we're inside a category, skip it.
            if self._state >= self._STATE_InsideCategory:
                self._skipCurrentCategory = True
            
    def _characters(self, ch):
        text = unicode(ch)
        if self._state == self._STATE_InsidePattern:
            # TODO: text inside patterns must be upper-case!
            self._currentPattern += text
        elif self._state == self._STATE_InsideThat:
            self._currentThat += text
        elif self._state == self._STATE_InsideTemplate:
            # First, see whether the element at the top of the element stack
            # is permitted to contain text.
            try:
                parent = self._elemStack[-1][0]
                parentAttr = self._elemStack[-1][1]
                required, optional, canBeParent = self._validInfo[parent]
                nonBlockStyleCondition = (parent == "condition" and not ("name" in parentAttr and "value" in parentAttr))
                if not canBeParent:
                    raise AimlParserError( ("Unexpected text inside <%s> element "%parent)+self._location() )
                elif parent == "random" or nonBlockStyleCondition:
                    # <random> elements can only contain <li> subelements. However,
                    # there's invariably some whitespace around the <li> that we need
                    # to ignore. Same for non-block-style <condition> elements (i.e.
                    # those which don't have both a "name" and a "value" attribute).
                    if len(text.strip()) == 0:
                        # ignore whitespace inside these elements.
                        return
                    else:
                        # non-whitespace text inside these elements is a syntax error.
                        raise AimlParserError( ("Unexpected text inside <%s> element "%parent)+self._location() )
            except IndexError:
                # the element stack is empty. This should never happen.
                raise AimlParserError( "Element stack is empty while validating text "+self._location() )
            
            # Add a new text element to the element at the top of the element
            # stack. If there's already a text element there, simply append the
            # new characters to its contents.
            try: textElemOnStack = (self._elemStack[-1][-1][0] == "text")
            except IndexError: textElemOnStack = False
            except KeyError: textElemOnStack = False
            if textElemOnStack:
                self._elemStack[-1][-1][2] += text
            else:
                self._elemStack[-1].append(["text", {"xml:space": self._whitespaceBehaviorStack[-1]}, text])
        else:
            # all other text is ignored
            pass

    def endElementNS(self, name, qname):
        uri, elem = name
        self.endElement(elem)
        
    def endElement(self, name):
        """Wrapper around _endElement which catches errors in _characters()
        and keeps going.
        """     
        if self._state == self._STATE_OutsideAiml:
            # If we're outside of an AIML element, ignore all tags
            return
        if self._currentUnknown != "":
            # see if we're at the end of an unknown element.  If so, we can
            # stop ignoring everything.
            if name == self._currentUnknown:
                self._currentUnknown = ""
            return
        if self._skipCurrentCategory:
            # If we're skipping the current category, see if it's ending. We
            # stop on ANY </category> tag, since we're not keeping track of
            # state in ignore-mode.
            if name == "category":
                self._skipCurrentCategory = False
                self._state = self._STATE_InsideAiml
            return
        try: self._endElement(name)
        except AimlParserError as msg:
            # Print the message
            sys.stderr.write("PARSE ERROR: %s\n" % msg)
            self._numParseErrors += 1 # increment error count
            # In case of a parse error, if we're inside a category, skip it.
            if self._state >= self._STATE_InsideCategory:
                self._skipCurrentCategory = True

    def _endElement(self, name):
        """
        Verify that an AIML end element is valid in the current context.
        Raises an AimlParserError if an illegal end element is encountered.
        """
        if name == "aiml":
            # </aiml> tags are only legal in the InsideAiml state
            if self._state != self._STATE_InsideAiml:
                raise AimlParserError( "Unexpected </aiml> tag "+self._location() )
            self._state = self._STATE_OutsideAiml
            self._whitespaceBehaviorStack.pop()
        elif name == "topic":
            # </topic> tags are only legal in the InsideAiml state, and
            # only if _insideTopic is true.
            if self._state != self._STATE_InsideAiml or not self._insideTopic:
                raise AimlParserError( "Unexpected </topic> tag "+self._location() )
            self._insideTopic = False
            self._currentTopic = u""
        elif name == "category":
            # </category> tags are only legal in the AfterTemplate state
            if self._state != self._STATE_AfterTemplate:
                raise AimlParserError( "Unexpected </category> tag "+self._location() )
            self._state = self._STATE_InsideAiml
            # End the current category.  Store the current pattern/that/topic and
            # element in the categories dictionary.
            key = (self._currentPattern.strip(), self._currentThat.strip(),self._currentTopic.strip())
            self.categories[key] = self._elemStack[-1]
            self._whitespaceBehaviorStack.pop()
        elif name == "pattern":
            # </pattern> tags are only legal in the InsidePattern state
            if self._state != self._STATE_InsidePattern:
                raise AimlParserError( "Unexpected </pattern> tag "+self._location() )
            self._state = self._STATE_AfterPattern
        elif name == "that" and self._state == self._STATE_InsideThat:
            # </that> tags are only allowed inside <template> elements or in
            # the InsideThat state.  This clause handles the latter case.
            self._state = self._STATE_AfterThat
        elif name == "template":
            # </template> tags are only allowed in the InsideTemplate state.
            if self._state != self._STATE_InsideTemplate:
                raise AimlParserError( "Unexpected </template> tag "+self._location() )
            self._state = self._STATE_AfterTemplate
            self._whitespaceBehaviorStack.pop()
        elif self._state == self._STATE_InsidePattern:
            # Certain tags are allowed inside <pattern> elements.
            if name not in ["bot"]:
                raise AimlParserError( ("Unexpected </%s> tag " % name)+self._location() )
        elif self._state == self._STATE_InsideThat:
            # Certain tags are allowed inside <that> elements.
            if name not in ["bot"]:
                raise AimlParserError( ("Unexpected </%s> tag " % name)+self._location() )
        elif self._state == self._STATE_InsideTemplate:
            # End of an element inside the current template.  Append the
            # element at the top of the stack onto the one beneath it.
            elem = self._elemStack.pop()
            self._elemStack[-1].append(elem)
            self._whitespaceBehaviorStack.pop()
            # If the element was a condition, pop an item off the
            # foundDefaultLiStack as well.
            if elem[0] == "condition": self._foundDefaultLiStack.pop()
        else:
            # Unexpected closing tag
            raise AimlParserError( ("Unexpected </%s> tag " % name)+self._location() )

    # A dictionary containing a validation information for each AIML
    # element. The keys are the names of the elements.  The values are a
    # tuple of three items. The first is a list containing the names of
    # REQUIRED attributes, the second is a list of OPTIONAL attributes,
    # and the third is a boolean value indicating whether or not the
    # element can contain other elements and/or text (if False, the
    # element can only appear in an atomic context, such as <date/>).
    _validationInfo101 = {
        "bot":          ( ["name"], [], False ),
        "condition":    ( [], ["name", "value"], True ), # can only contain <li> elements
        "date":         ( [], [], False ),
        "formal":       ( [], [], True ),
        "gender":       ( [], [], True ),
        "get":          ( ["name"], [], False ),
        "gossip":       ( [], [], True ),
        "id":           ( [], [], False ),
        "input":        ( [], ["index"], False ),
        "javascript":   ( [], [], True ),
        "learn":        ( [], [], True ),
        "li":           ( [], ["name", "value"], True ),
        "lowercase":    ( [], [], True ),
        "person":       ( [], [], True ),
        "person2":      ( [], [], True ),
        "random":       ( [], [], True ), # can only contain <li> elements
        "sentence":     ( [], [], True ),
        "set":          ( ["name"], [], True),
        "size":         ( [], [], False ),
        "sr":           ( [], [], False ),
        "srai":         ( [], [], True ),
        "star":         ( [], ["index"], False ),
        "system":       ( [], [], True ),
        "template":     ( [], [], True ), # needs to be in the list because it can be a parent.
        "that":         ( [], ["index"], False ),
        "thatstar":     ( [], ["index"], False ),
        "think":        ( [], [], True ),
        "topicstar":    ( [], ["index"], False ),
        "uppercase":    ( [], [], True ),
        "version":      ( [], [], False ),
    }

    def _validateElemStart(self, name, attr, version):
        """
        Test the validity of an element starting inside a <template> element.

        This function raises an AimlParserError exception if it the tag is
        invalid.  Otherwise, no news is good news.
        """     
        # Check the element's attributes.  Make sure that all required
        # attributes are present, and that any remaining attributes are
        # valid options.        
        required, optional, canBeParent = self._validInfo[name]
        for a in required:
            if a not in attr and not self._forwardCompatibleMode:
                raise AimlParserError( ("Required \"%s\" attribute missing in <%s> element " % (a,name))+self._location() )
        for a in attr:
            if a in required: continue
            if a[0:4] == "xml:": continue # attributes in the "xml" namespace can appear anywhere
            if a not in optional and not self._forwardCompatibleMode:
                raise AimlParserError( ("Unexpected \"%s\" attribute in <%s> element " % (a,name))+self._location() )

        # special-case: several tags contain an optional "index" attribute.
        # This attribute's value must be a positive integer.
        if name in ["star", "thatstar", "topicstar"]:
            for k,v in attr.items():
                if k == "index":
                    temp = 0
                    try: temp = int(v)
                    except:
                        raise AimlParserError( ("Bad type for \"%s\" attribute (expected integer, found \"%s\") " % (k,v))+self._location() )
                    if temp < 1:
                        raise AimlParserError( ("\"%s\" attribute must have non-negative value " % (k))+self._location() )

        # See whether the containing element is permitted to contain
        # subelements. If not, this element is invalid no matter what it is.
        try:
            parent = self._elemStack[-1][0]
            parentAttr = self._elemStack[-1][1]
        except IndexError:
            # If the stack is empty, no parent is present.  This should never
            # happen.
            raise AimlParserError( ("Element stack is empty while validating <%s> " % name)+self._location() )
        required, optional, canBeParent = self._validInfo[parent]
        nonBlockStyleCondition = (parent == "condition" and not ("name" in parentAttr and "value" in parentAttr))
        if not canBeParent:
            raise AimlParserError( ("<%s> elements cannot have any contents "%parent)+self._location() )
        # Special-case test if the parent element is <condition> (the
        # non-block-style variant) or <random>: these elements can only
        # contain <li> subelements.
        elif (parent == "random" or nonBlockStyleCondition) and name!="li":
            raise AimlParserError( ("<%s> elements can only contain <li> subelements "%parent)+self._location() )
        # Special-case test for <li> elements, which can only be contained
        # by non-block-style <condition> and <random> elements, and whose
        # required attributes are dependent upon which attributes are
        # present in the <condition> parent.
        elif name=="li":
            if not (parent=="random" or nonBlockStyleCondition):
                raise AimlParserError( ("Unexpected <li> element contained by <%s> element "%parent)+self._location() )
            if nonBlockStyleCondition:
                if "name" in parentAttr:
                    # Single-predicate condition.  Each <li> element except the
                    # last must have a "value" attribute.
                    if len(attr) == 0:
                        # This could be the default <li> element for this <condition>,
                        # unless we've already found one.
                        if self._foundDefaultLiStack[-1]:
                            raise AimlParserError( "Unexpected default <li> element inside <condition> "+self._location() )
                        else:
                            self._foundDefaultLiStack[-1] = True
                    elif len(attr) == 1 and "value" in attr:
                        pass # this is the valid case
                    else:
                        raise AimlParserError( "Invalid <li> inside single-predicate <condition> "+self._location() )
                elif len(parentAttr) == 0:
                    # Multi-predicate condition.  Each <li> element except the
                    # last must have a "name" and a "value" attribute.
                    if len(attr) == 0:
                        # This could be the default <li> element for this <condition>,
                        # unless we've already found one.
                        if self._foundDefaultLiStack[-1]:
                            raise AimlParserError( "Unexpected default <li> element inside <condition> "+self._location() )
                        else:
                            self._foundDefaultLiStack[-1] = True
                    elif len(attr) == 2 and "value" in attr and "name" in attr:
                        pass # this is the valid case
                    else:
                        raise AimlParserError( "Invalid <li> inside multi-predicate <condition> "+self._location() )
        # All is well!
        return True

def create_parser():
    """Create and return an AIML parser object."""
    parser = xml.sax.make_parser()
    handler = AimlHandler("UTF-8")
    parser.setContentHandler(handler)
    #parser.setFeature(xml.sax.handler.feature_namespaces, True)
    return parser
