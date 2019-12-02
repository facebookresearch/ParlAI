# -*- coding: latin-1 -*-
"""This file contains the public interface to the aiml module."""

from __future__ import print_function

import copy
import glob
import os
import random
import re
import string
import sys
import time
import threading
import xml.sax
from collections import namedtuple
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

from .constants import *
from . import DefaultSubs
from . import Utils
from .AimlParser import create_parser
from .PatternMgr import PatternMgr
from .WordSub import WordSub



def msg_encoder(encoding=None):
    """
    Return a named tuple with a pair of functions to encode/decode messages.
    For None encoding, a passthrough function will be returned
    """
    Codec = namedtuple('Codec', ['enc','dec'])
    if encoding in (None, False):
        l = lambda x: unicode(x)
        return Codec(l, l)
    else:
        return Codec(lambda x: x.encode(encoding, 'replace'),
                     lambda x: x.decode(encoding, 'replace'))




class Kernel:
    # module constants
    _globalSessionID = "_global" # key of the global session (duh)
    _maxHistorySize = 10 # maximum length of the _inputs and _responses lists
    _maxRecursionDepth = 100 # maximum number of recursive <srai>/<sr> tags before the response is aborted.
    # special predicate keys
    _inputHistory = "_inputHistory"     # keys to a queue (list) of recent user input
    _outputHistory = "_outputHistory"   # keys to a queue (list) of recent responses.
    _inputStack = "_inputStack"         # Should always be empty in between calls to respond()

    def __init__(self):
        self._verboseMode = True
        self._version = "python-aiml {}".format(VERSION)
        self._brain = PatternMgr()
        self._respondLock = threading.RLock()
        self.setTextEncoding(None if PY3 else "utf-8")

        # set up the sessions
        self._sessions = {}
        self._addSession(self._globalSessionID)

        # Set up the bot predicates
        self._botPredicates = {}
        self.setBotPredicate("name", "Nameless")

        # set up the word substitutors (subbers):
        self._subbers = {}
        self._subbers['gender'] = WordSub(DefaultSubs.defaultGender)
        self._subbers['person'] = WordSub(DefaultSubs.defaultPerson)
        self._subbers['person2'] = WordSub(DefaultSubs.defaultPerson2)
        self._subbers['normal'] = WordSub(DefaultSubs.defaultNormal)

        # set up the element processors
        self._elementProcessors = {
            "bot":          self._processBot,
            "condition":    self._processCondition,
            "date":         self._processDate,
            "formal":       self._processFormal,
            "gender":       self._processGender,
            "get":          self._processGet,
            "gossip":       self._processGossip,
            "id":           self._processId,
            "input":        self._processInput,
            "javascript":   self._processJavascript,
            "learn":        self._processLearn,
            "li":           self._processLi,
            "lowercase":    self._processLowercase,
            "person":       self._processPerson,
            "person2":      self._processPerson2,
            "random":       self._processRandom,
            "text":         self._processText,
            "sentence":     self._processSentence,
            "set":          self._processSet,
            "size":         self._processSize,
            "sr":           self._processSr,
            "srai":         self._processSrai,
            "star":         self._processStar,
            "system":       self._processSystem,
            "template":     self._processTemplate,
            "that":         self._processThat,
            "thatstar":     self._processThatstar,
            "think":        self._processThink,
            "topicstar":    self._processTopicstar,
            "uppercase":    self._processUppercase,
            "version":      self._processVersion,
        }

    def bootstrap(self, brainFile=None, learnFiles=[], commands=[],
                  chdir=None):
        """Prepare a Kernel object for use.

        If a `brainFile` argument is provided, the Kernel attempts to
        load the brain at the specified filename.

        If `learnFiles` is provided, the Kernel attempts to load the
        specified AIML files.

        Finally, each of the input strings in the `commands` list is
        passed to respond().

        The `chdir` argument makes the it change to that directory before
        performing any learn or command execution (but after loadBrain
        processing). Upon returning the current directory is moved back to
        where it was before.
        """
        start = time.time()
        if brainFile:
            self.loadBrain(brainFile)

        prev = os.getcwd()
        try:
            if chdir:
                os.chdir(chdir)

            # learnFiles might be a string, in which case it should be
            # turned into a single-element list.
            if isinstance(learnFiles, (str, unicode)):
                learnFiles = (learnFiles,)
            for file in learnFiles:
                self.learn(file)

            # ditto for commands
            if isinstance(commands, (str, unicode)):
                commands = (commands,)
            for cmd in commands:
                print(self._respond(cmd, self._globalSessionID))

        finally:
            if chdir:
                os.chdir(prev)

        if self._verboseMode:
            print("Kernel bootstrap completed in %.2f seconds" % (time.time() - start))

    def verbose(self, isVerbose=True):
        """Enable/disable verbose output mode."""
        self._verboseMode = isVerbose

    def version(self):
        """Return the Kernel's version string."""
        return self._version

    def numCategories(self):
        """Return the number of categories the Kernel has learned."""
        # there's a one-to-one mapping between templates and categories
        return self._brain.numTemplates()

    def resetBrain(self):
        """Reset the brain to its initial state.

        This is essentially equivilant to:
            del(kern)
            kern = aiml.Kernel()

        """
        del(self._brain)
        self.__init__()

    def loadBrain(self, filename):
        """Attempt to load a previously-saved 'brain' from the
        specified filename.

        NOTE: the current contents of the 'brain' will be discarded!

        """
        if self._verboseMode: print( "Loading brain from %s..." % filename, end="" )
        start = time.time()
        self._brain.restore(filename)
        if self._verboseMode:
            end = time.time() - start
            print( "done (%d categories in %.2f seconds)" % (self._brain.numTemplates(), end) )

    def saveBrain(self, filename):
        """Dump the contents of the bot's brain to a file on disk."""
        if self._verboseMode: print( "Saving brain to %s..." % filename, end="")
        start = time.time()
        self._brain.save(filename)
        if self._verboseMode:
            print("done (%.2f seconds)" % (time.time() - start))

    def getPredicate(self, name, sessionID=_globalSessionID):
        """Retrieve the current value of the predicate 'name' from the
        specified session.

        If name is not a valid predicate in the session, the empty
        string is returned.

        """
        try: return self._sessions[sessionID][name]
        except KeyError: return ""

    def setPredicate(self, name, value, sessionID = _globalSessionID):
        """Set the value of the predicate 'name' in the specified
        session.

        If sessionID is not a valid session, it will be created. If
        name is not a valid predicate in the session, it will be
        created.

        """
        self._addSession(sessionID)  # add the session, if it doesn't already exist.
        self._sessions[sessionID][name] = value

    def getBotPredicate(self, name):
        """Retrieve the value of the specified bot predicate.

        If name is not a valid bot predicate, the empty string is returned.

        """
        try: return self._botPredicates[name]
        except KeyError: return ""

    def setBotPredicate(self, name, value):
        """Set the value of the specified bot predicate.

        If name is not a valid bot predicate, it will be created.

        """
        self._botPredicates[name] = value
        # Clumsy hack: if updating the bot name, we must update the
        # name in the brain as well
        if name == "name":
            self._brain.setBotName(self.getBotPredicate("name"))

    def setTextEncoding(self, encoding):
        """
        Set the I/O text encoding expected. All strings loaded from AIML files
        will be converted to it. 
        The respond() method is expected to be passed strings encoded with it 
        (str in Py2, bytes in Py3) and will also return them.
        If it is False, then strings are assumed *not* to be encoded, i.e.
        they will be unicode strings (unicode in Py2, str in Py3)
        """
        self._textEncoding = encoding
        self._cod = msg_encoder(encoding)


    def loadSubs(self, filename):
        """Load a substitutions file.

        The file must be in the Windows-style INI format (see the
        standard ConfigParser module docs for information on this
        format).  Each section of the file is loaded into its own
        substituter.

        """
        parser = ConfigParser()
        with open(filename) as inFile:
            parser.readfp(inFile, filename)
        for s in parser.sections():
            # Add a new WordSub instance for this section.  If one already
            # exists, delete it.
            if s in self._subbers:
                del(self._subbers[s])
            self._subbers[s] = WordSub()
            # iterate over the key,value pairs and add them to the subber
            for k, v in parser.items(s):
                self._subbers[s][k] = v

    def _addSession(self, sessionID):
        """Create a new session with the specified ID string."""
        if sessionID in self._sessions:
            return
        # Create the session.
        self._sessions[sessionID] = {
            # Initialize the special reserved predicates
            self._inputHistory: [],
            self._outputHistory: [],
            self._inputStack: []
        }

    def _deleteSession(self, sessionID):
        """Delete the specified session."""
        if sessionID in self._sessions:
            self._sessions.pop(sessionID)

    def getSessionData(self, sessionID=None):
        """Return a copy of the session data dictionary for the
        specified session.

        If no sessionID is specified, return a dictionary containing
        *all* of the individual session dictionaries.

        """
        s = None
        if sessionID is not None:
            try: s = self._sessions[sessionID]
            except KeyError: s = {}
        else:
            s = self._sessions
        return copy.deepcopy(s)

    def learn(self, filename):
        """Load and learn the contents of the specified AIML file.

        If filename includes wildcard characters, all matching files
        will be loaded and learned.

        """
        for f in glob.glob(filename):
            if self._verboseMode: print( "Loading %s..." % f, end="")
            start = time.time()
            # Load and parse the AIML file.
            parser = create_parser()
            handler = parser.getContentHandler()
            handler.setEncoding(self._textEncoding)
            try: parser.parse(f)
            except xml.sax.SAXParseException as msg:
                err = "\nFATAL PARSE ERROR in file %s:\n%s\n" % (f,msg)
                sys.stderr.write(err)
                continue
            # store the pattern/template pairs in the PatternMgr.
            for key, tem in handler.categories.items():
                self._brain.add(key, tem)
            # Parsing was successful.
            if self._verboseMode:
                print("done (%.2f seconds)" % (time.time() - start))

    def respond(self, input_, sessionID=_globalSessionID):
        """Return the Kernel's response to the input string."""
        if len(input_) == 0:
            return u""

        # Decode the input (assumed to be an encoded string) into a unicode
        # string. Note that if encoding is False, this will be a no-op
        try: input_ = self._cod.dec(input_)
        except UnicodeError: pass
        except AttributeError: pass

        # prevent other threads from stomping all over us.
        self._respondLock.acquire()

        try:
            # Add the session, if it doesn't already exist
            self._addSession(sessionID)

            # split the input into discrete sentences
            sentences = Utils.sentences(input_)
            finalResponse = u""
            for s in sentences:
                # Add the input to the history list before fetching the
                # response, so that <input/> tags work properly.
                inputHistory = self.getPredicate(self._inputHistory, sessionID)
                inputHistory.append(s)
                while len(inputHistory) > self._maxHistorySize:
                    inputHistory.pop(0)
                self.setPredicate(self._inputHistory, inputHistory, sessionID)

                # Fetch the response
                response = self._respond(s, sessionID)

                # add the data from this exchange to the history lists
                outputHistory = self.getPredicate(self._outputHistory, sessionID)
                outputHistory.append(response)
                while len(outputHistory) > self._maxHistorySize:
                    outputHistory.pop(0)
                self.setPredicate(self._outputHistory, outputHistory, sessionID)

                # append this response to the final response.
                finalResponse += (response + u"  ")

            finalResponse = finalResponse.strip()
            #print( "@ASSERT", self.getPredicate(self._inputStack, sessionID))
            assert(len(self.getPredicate(self._inputStack, sessionID)) == 0)

            # and return, encoding the string into the I/O encoding
            return self._cod.enc(finalResponse)

        finally:
            # release the lock
            self._respondLock.release()


    # This version of _respond() just fetches the response for some input.
    # It does not mess with the input and output histories.  Recursive calls
    # to respond() spawned from tags like <srai> should call this function
    # instead of respond().
    def _respond(self, input_, sessionID):
        """Private version of respond(), does the real work."""
        if len(input_) == 0:
            return u""

        # guard against infinite recursion
        inputStack = self.getPredicate(self._inputStack, sessionID)
        if len(inputStack) > self._maxRecursionDepth:
            if self._verboseMode:
                err = u"WARNING: maximum recursion depth exceeded (input='%s')" % self._cod.enc(input_)
                sys.stderr.write(err)
            return u""

        # push the input onto the input stack
        inputStack = self.getPredicate(self._inputStack, sessionID)
        inputStack.append(input_)
        self.setPredicate(self._inputStack, inputStack, sessionID)

        # run the input through the 'normal' subber
        subbedInput = self._subbers['normal'].sub(input_)

        # fetch the bot's previous response, to pass to the match()
        # function as 'that'.
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        try: that = outputHistory[-1]
        except IndexError: that = ""
        subbedThat = self._subbers['normal'].sub(that)

        # fetch the current topic
        topic = self.getPredicate("topic", sessionID)
        subbedTopic = self._subbers['normal'].sub(topic)

        # Determine the final response.
        response = u""
        elem = self._brain.match(subbedInput, subbedThat, subbedTopic)
        if elem is None:
            if self._verboseMode:
                err = "WARNING: No match found for input: %s\n" % self._cod.enc(input_)
                sys.stderr.write(err)
        else:
            # Process the element into a response string.
            response += self._processElement(elem, sessionID).strip()
            response += u" "
        response = response.strip()

        # pop the top entry off the input stack.
        inputStack = self.getPredicate(self._inputStack, sessionID)
        inputStack.pop()
        self.setPredicate(self._inputStack, inputStack, sessionID)

        return response

    def _processElement(self, elem, sessionID):
        """Process an AIML element.

        The first item of the elem list is the name of the element's
        XML tag.  The second item is a dictionary containing any
        attributes passed to that tag, and their values.  Any further
        items in the list are the elements enclosed by the current
        element's begin and end tags; they are handled by each
        element's handler function.

        """
        try:
            handlerFunc = self._elementProcessors[elem[0]]
        except Exception:
            # Oops -- there's no handler function for this element type!
            if self._verboseMode:
                err = "WARNING: No handler found for <%s> element\n" % self._cod.enc(elem[0])
                sys.stderr.write(err)
            return u""
        return handlerFunc(elem, sessionID)


    ######################################################
    ### Individual element-processing functions follow ###
    ######################################################

    # <bot>
    def _processBot(self, elem, sessionID):
        """Process a <bot> AIML element.

        Required element attributes:
            name: The name of the bot predicate to retrieve.

        <bot> elements are used to fetch the value of global,
        read-only "bot predicates."  These predicates cannot be set
        from within AIML; you must use the setBotPredicate() function.

        """
        attrName = elem[1]['name']
        return self.getBotPredicate(attrName)

    # <condition>
    def _processCondition(self, elem, sessionID):
        """Process a <condition> AIML element.

        Optional element attributes:
            name: The name of a predicate to test.
            value: The value to test the predicate for.

        <condition> elements come in three flavors.  Each has different
        attributes, and each handles their contents differently.

        The simplest case is when the <condition> tag has both a 'name'
        and a 'value' attribute.  In this case, if the predicate
        'name' has the value 'value', then the contents of the element
        are processed and returned.

        If the <condition> element has only a 'name' attribute, then
        its contents are a series of <li> elements, each of which has
        a 'value' attribute.  The list is scanned from top to bottom
        until a match is found.  Optionally, the last <li> element can
        have no 'value' attribute, in which case it is processed and
        returned if no other match is found.

        If the <condition> element has neither a 'name' nor a 'value'
        attribute, then it behaves almost exactly like the previous
        case, except that each <li> subelement (except the optional
        last entry) must now include both 'name' and 'value'
        attributes.

        """
        attr = None
        response = ""
        attr = elem[1]

        # Case #1: test the value of a specific predicate for a
        # specific value.
        if 'name' in attr and 'value' in attr:
            val = self.getPredicate(attr['name'], sessionID)
            if val == attr['value']:
                for e in elem[2:]:
                    response += self._processElement(e, sessionID)
                return response
        else:
            # Case #2 and #3: Cycle through <li> contents, testing a
            # name and value pair for each one.
            try:
                name = attr.get('name', None)
                # Get the list of <li> elemnents
                listitems = []
                for e in elem[2:]:
                    if e[0] == 'li':
                        listitems.append(e)
                # if listitems is empty, return the empty string
                if len(listitems) == 0:
                    return ""
                # iterate through the list looking for a condition that
                # matches.
                foundMatch = False
                for li in listitems:
                    try:
                        liAttr = li[1]
                        # if this is the last list item, it's allowed
                        # to have no attributes.  We just skip it for now.
                        if len(liAttr) == 0 and li == listitems[-1]:
                            continue
                        # get the name of the predicate to test
                        liName = name
                        if liName is None:
                            liName = liAttr['name']
                        # get the value to check against
                        liValue = liAttr['value']
                        # do the test
                        if self.getPredicate(liName, sessionID) == liValue:
                            foundMatch = True
                            response += self._processElement(li, sessionID)
                            break
                    except Exception:
                        # No attributes, no name/value attributes, no
                        # such predicate/session, or processing error.
                        if self._verboseMode: print("Something amiss -- skipping listitem", li)
                        raise
                if not foundMatch:
                    # Check the last element of listitems.  If it has
                    # no 'name' or 'value' attribute, process it.
                    try:
                        li = listitems[-1]
                        liAttr = li[1]
                        if not ('name' in liAttr or 'value' in liAttr):
                            response += self._processElement(li, sessionID)
                    except Exception:
                        # listitems was empty, no attributes, missing
                        # name/value attributes, or processing error.
                        if self._verboseMode: print("error in default listitem")
                        raise
            except Exception:
                # Some other catastrophic cataclysm
                if self._verboseMode: print("catastrophic condition failure")
                raise
        return response

    # <date>
    def _processDate(self, elem, sessionID):
        """Process a <date> AIML element.

        <date> elements resolve to the current date and time.  The
        AIML specification doesn't require any particular format for
        this information, so I go with whatever's simplest.

        """
        return time.asctime()

    # <formal>
    def _processFormal(self, elem, sessionID):
        """Process a <formal> AIML element.

        <formal> elements process their contents recursively, and then
        capitalize the first letter of each word of the result.

        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return string.capwords(response)

    # <gender>
    def _processGender(self, elem, sessionID):
        """Process a <gender> AIML element.

        <gender> elements process their contents, and then swap the
        gender of any third-person singular pronouns in the result.
        This subsitution is handled by the aiml.WordSub module.

        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return self._subbers['gender'].sub(response)

    # <get>
    def _processGet(self, elem, sessionID):
        """Process a <get> AIML element.

        Required element attributes:
            name: The name of the predicate whose value should be
            retrieved from the specified session and returned.  If the
            predicate doesn't exist, the empty string is returned.

        <get> elements return the value of a predicate from the
        specified session.

        """
        return self.getPredicate(elem[1]['name'], sessionID)

    # <gossip>
    def _processGossip(self, elem, sessionID):
        """Process a <gossip> AIML element.

        <gossip> elements are used to capture and store user input in
        an implementation-defined manner, theoretically allowing the
        bot to learn from the people it chats with.  I haven't
        descided how to define my implementation, so right now
        <gossip> behaves identically to <think>.

        """
        return self._processThink(elem, sessionID)

    # <id>
    def _processId(self, elem, sessionID):
        """ Process an <id> AIML element.

        <id> elements return a unique "user id" for a specific
        conversation.  In PyAIML, the user id is the name of the
        current session.

        """
        return sessionID

    # <input>
    def _processInput(self, elem, sessionID):
        """Process an <input> AIML element.

        Optional attribute elements:
            index: The index of the element from the history list to
            return. 1 means the most recent item, 2 means the one
            before that, and so on.

        <input> elements return an entry from the input history for
        the current session.

        """
        inputHistory = self.getPredicate(self._inputHistory, sessionID)
        try: index = int(elem[1]['index'])
        except: index = 1
        try: return inputHistory[-index]
        except IndexError:
            if self._verboseMode:
                err = "No such index %d while processing <input> element.\n" % index
                sys.stderr.write(err)
            return ""

    # <javascript>
    def _processJavascript(self, elem, sessionID):
        """Process a <javascript> AIML element.

        <javascript> elements process their contents recursively, and
        then run the results through a server-side Javascript
        interpreter to compute the final response.  Implementations
        are not required to provide an actual Javascript interpreter,
        and right now PyAIML doesn't; <javascript> elements are behave
        exactly like <think> elements.

        """
        return self._processThink(elem, sessionID)

    # <learn>
    def _processLearn(self, elem, sessionID):
        """Process a <learn> AIML element.

        <learn> elements process their contents recursively, and then
        treat the result as an AIML file to open and learn.

        """
        filename = ""
        for e in elem[2:]:
            filename += self._processElement(e, sessionID)
        self.learn(filename)
        return ""

    # <li>
    def _processLi(self, elem, sessionID):
        """Process an <li> AIML element.

        Optional attribute elements:
            name: the name of a predicate to query.
            value: the value to check that predicate for.

        <li> elements process their contents recursively and return
        the results. They can only appear inside <condition> and
        <random> elements.  See _processCondition() and
        _processRandom() for details of their usage.

        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return response

    # <lowercase>
    def _processLowercase(self, elem, sessionID):
        """Process a <lowercase> AIML element.

        <lowercase> elements process their contents recursively, and
        then convert the results to all-lowercase.

        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return response.lower()

    # <person>
    def _processPerson(self, elem, sessionID):
        """Process a <person> AIML element.

        <person> elements process their contents recursively, and then
        convert all pronouns in the results from 1st person to 2nd
        person, and vice versa.  This subsitution is handled by the
        aiml.WordSub module.

        If the <person> tag is used atomically (e.g. <person/>), it is
        a shortcut for <person><star/></person>.

        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        if len(elem[2:]) == 0:  # atomic <person/> = <person><star/></person>
            response = self._processElement(['star', {}], sessionID)
        return self._subbers['person'].sub(response)

    # <person2>
    def _processPerson2(self, elem, sessionID):
        """Process a <person2> AIML element.

        <person2> elements process their contents recursively, and then
        convert all pronouns in the results from 1st person to 3rd
        person, and vice versa.  This subsitution is handled by the
        aiml.WordSub module.

        If the <person2> tag is used atomically (e.g. <person2/>), it is
        a shortcut for <person2><star/></person2>.

        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        if len(elem[2:]) == 0:  # atomic <person2/> = <person2><star/></person2>
            response = self._processElement(['star', {}], sessionID)
        return self._subbers['person2'].sub(response)

    # <random>
    def _processRandom(self, elem, sessionID):
        """Process a <random> AIML element.

        <random> elements contain zero or more <li> elements.  If
        none, the empty string is returned.  If one or more <li>
        elements are present, one of them is selected randomly to be
        processed recursively and have its results returned.  Only the
        chosen <li> element's contents are processed.  Any non-<li> contents are
        ignored.

        """
        listitems = []
        for e in elem[2:]:
            if e[0] == 'li':
                listitems.append(e)
        if len(listitems) == 0:
            return ""

        # select and process a random listitem.
        random.shuffle(listitems)
        return self._processElement(listitems[0], sessionID)

    # <sentence>
    def _processSentence(self, elem, sessionID):
        """Process a <sentence> AIML element.

        <sentence> elements process their contents recursively, and
        then capitalize the first letter of the results.

        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        try:
            response = response.strip()
            words = response.split(" ", 1)
            words[0] = words[0].capitalize()
            response = ' '.join(words)
            return response
        except IndexError:  # response was empty
            return ""

    # <set>
    def _processSet(self, elem, sessionID):
        """Process a <set> AIML element.

        Required element attributes:
            name: The name of the predicate to set.

        <set> elements process their contents recursively, and assign the results to a predicate
        (given by their 'name' attribute) in the current session.  The contents of the element
        are also returned.

        """
        value = ""
        for e in elem[2:]:
            value += self._processElement(e, sessionID)
        #print( "@ELEM", elem )
        self.setPredicate(elem[1]['name'], value, sessionID)
        return value

    # <size>
    def _processSize(self, elem, sessionID):
        """Process a <size> AIML element.

        <size> elements return the number of AIML categories currently
        in the bot's brain.

        """
        return str(self.numCategories())

    # <sr>
    def _processSr(self, elem,sessionID):
        """Process an <sr> AIML element.

        <sr> elements are shortcuts for <srai><star/></srai>.

        """
        star = self._processElement(['star', {}], sessionID)
        response = self._respond(star, sessionID)
        return response

    # <srai>
    def _processSrai(self, elem, sessionID):
        """Process a <srai> AIML element.

        <srai> elements recursively process their contents, and then
        pass the results right back into the AIML interpreter as a new
        piece of input.  The results of this new input string are
        returned.

        """
        newInput = ""
        for e in elem[2:]:
            newInput += self._processElement(e, sessionID)
        return self._respond(newInput, sessionID)

    # <star>
    def _processStar(self, elem, sessionID):
        """Process a <star> AIML element.

        Optional attribute elements:
            index: Which "*" character in the current pattern should
            be matched?

        <star> elements return the text fragment matched by the "*"
        character in the current input pattern.  For example, if the
        input "Hello Tom Smith, how are you?" matched the pattern
        "HELLO * HOW ARE YOU", then a <star> element in the template
        would evaluate to "Tom Smith".

        """
        try: index = int(elem[1]['index'])
        except KeyError: index = 1
        # fetch the user's last input
        inputStack = self.getPredicate(self._inputStack, sessionID)
        input_ = self._subbers['normal'].sub(inputStack[-1])
        # fetch the Kernel's last response (for 'that' context)
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        try: that = self._subbers['normal'].sub(outputHistory[-1])
        except: that = ""  # there might not be any output yet
        topic = self.getPredicate("topic", sessionID)
        response = self._brain.star("star", input_, that, topic, index)
        return response

    # <system>
    def _processSystem(self, elem, sessionID):
        """Process a <system> AIML element.

        <system> elements process their contents recursively, and then
        attempt to execute the results as a shell command on the
        server.  The AIML interpreter blocks until the command is
        complete, and then returns the command's output.

        For cross-platform compatibility, any file paths inside
        <system> tags should use Unix-style forward slashes ("/") as a
        directory separator.

        """
        # build up the command string
        command = ""
        for e in elem[2:]:
            command += self._processElement(e, sessionID)

        # normalize the path to the command.  Under Windows, this
        # switches forward-slashes to back-slashes; all system
        # elements should use unix-style paths for cross-platform
        # compatibility.
        #executable,args = command.split(" ", 1)
        #executable = os.path.normpath(executable)
        #command = executable + " " + args
        command = os.path.normpath(command)

        # execute the command.
        response = ""
        try:
            out = os.popen(command)
        except RuntimeError as msg:
            if self._verboseMode:
                err = "WARNING: RuntimeError while processing \"system\" element:\n%s\n" % self._cod.enc(msg)
                sys.stderr.write(err)
            return "There was an error while computing my response.  Please inform my botmaster."
        time.sleep(0.01) # I'm told this works around a potential IOError exception.
        for line in out:
            response += line + "\n"
        response = ' '.join(response.splitlines()).strip()
        return response

    # <template>
    def _processTemplate(self, elem, sessionID):
        """Process a <template> AIML element.

        <template> elements recursively process their contents, and
        return the results.  <template> is the root node of any AIML
        response tree.

        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return response

    # text
    def _processText(self, elem, sessionID):
        """Process a raw text element.

        Raw text elements aren't really AIML tags. Text elements cannot contain
        other elements; instead, the third item of the 'elem' list is a text
        string, which is immediately returned. They have a single attribute,
        automatically inserted by the parser, which indicates whether whitespace
        in the text should be preserved or not.

        """
        try:
            elem[2] + ""
        except TypeError:
            raise TypeError("Text element contents are not text")

        # If the the whitespace behavior for this element is "default",
        # we reduce all stretches of >1 whitespace characters to a single
        # space.  To improve performance, we do this only once for each
        # text element encountered, and save the results for the future.
        if elem[1]["xml:space"] == "default":
            elem[2] = re.sub(r"\s+", " ", elem[2])
            elem[1]["xml:space"] = "preserve"
        return elem[2]

    # <that>
    def _processThat(self, elem, sessionID):
        """Process a <that> AIML element.

        Optional element attributes:
            index: Specifies which element from the output history to
            return.  1 is the most recent response, 2 is the next most
            recent, and so on.

        <that> elements (when they appear inside <template> elements)
        are the output equivilant of <input> elements; they return one
        of the Kernel's previous responses.

        """
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        index = 1
        try:
            # According to the AIML spec, the optional index attribute
            # can either have the form "x" or "x,y". x refers to how
            # far back in the output history to go.  y refers to which
            # sentence of the specified response to return.
            index = int(elem[1]['index'].split(',')[0])
        except Exception:
            pass
        try: return outputHistory[-index]
        except IndexError:
            if self._verboseMode:
                err = "No such index %d while processing <that> element.\n" % index
                sys.stderr.write(err)
            return ""

    # <thatstar>
    def _processThatstar(self, elem, sessionID):
        """Process a <thatstar> AIML element.

        Optional element attributes:
            index: Specifies which "*" in the <that> pattern to match.

        <thatstar> elements are similar to <star> elements, except
        that where <star/> returns the portion of the input string
        matched by a "*" character in the pattern, <thatstar/> returns
        the portion of the previous input string that was matched by a
        "*" in the current category's <that> pattern.

        """
        try: index = int(elem[1]['index'])
        except KeyError: index = 1
        # fetch the user's last input
        inputStack = self.getPredicate(self._inputStack, sessionID)
        input_ = self._subbers['normal'].sub(inputStack[-1])
        # fetch the Kernel's last response (for 'that' context)
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        try: that = self._subbers['normal'].sub(outputHistory[-1])
        except Exception: that = ""  # there might not be any output yet
        topic = self.getPredicate("topic", sessionID)
        response = self._brain.star("thatstar", input_, that, topic, index)
        return response

    # <think>
    def _processThink(self, elem, sessionID):
        """Process a <think> AIML element.

        <think> elements process their contents recursively, and then
        discard the results and return the empty string.  They're
        useful for setting predicates and learning AIML files without
        generating any output.

        """
        for e in elem[2:]:
            self._processElement(e, sessionID)
        return ""

    # <topicstar>
    def _processTopicstar(self, elem, sessionID):
        """Process a <topicstar> AIML element.

        Optional element attributes:
            index: Specifies which "*" in the <topic> pattern to match.

        <topicstar> elements are similar to <star> elements, except
        that where <star/> returns the portion of the input string
        matched by a "*" character in the pattern, <topicstar/>
        returns the portion of current topic string that was matched
        by a "*" in the current category's <topic> pattern.

        """
        try: index = int(elem[1]['index'])
        except KeyError: index = 1
        # fetch the user's last input
        inputStack = self.getPredicate(self._inputStack, sessionID)
        input_ = self._subbers['normal'].sub(inputStack[-1])
        # fetch the Kernel's last response (for 'that' context)
        outputHistory = self.getPredicate(self._outputHistory, sessionID)
        try: that = self._subbers['normal'].sub(outputHistory[-1])
        except Exception: that = ""  # there might not be any output yet
        topic = self.getPredicate("topic", sessionID)
        response = self._brain.star("topicstar", input_, that, topic, index)
        return response

    # <uppercase>
    def _processUppercase(self, elem, sessionID):
        """Process an <uppercase> AIML element.

        <uppercase> elements process their contents recursively, and
        return the results with all lower-case characters converted to
        upper-case.

        """
        response = ""
        for e in elem[2:]:
            response += self._processElement(e, sessionID)
        return response.upper()

    # <version>
    def _processVersion(self, elem, sessionID):
        """Process a <version> AIML element.

        <version> elements return the version number of the AIML
        interpreter.

        """
        return self.version()
