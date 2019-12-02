python-aiml
***********

This is a fork of the `PyAIML`_ Python AIML interpreter. It has been
refactored to make it install and work in both Python 2.7 and Python 3.

PyAIML is (c) Cort Stratton. *python-aiml* uses the same license as PyAIML 
(2-clause BSD), except for the ALICE AIML files taken from the `Free ALICE AIML
set`_, which are licensed with the `LGPL`_ license.


Scripts
=======

Two small scripts are added upon installation:

* ``aiml-validate`` can be used to validate AIML files
* ``aiml-bot`` can be used to start a simple interactive session with a bot,
  after loading either AIML files or a saved brain file.


Datasets
========

The installation includes two AIML datasets:

* The *standard* AIML set, as it was included in PyAIML
* The `Free ALICE AIML set`_ v. 1.9, taken from the data published by the
  `ALICE AI Foundation`_ (with a few small fixes in files that did not 
  validate as `AIML 1.0.1`_)

They can be loaded via the ``bootstrap`` method in the ``Kernel`` class. See 
the `bot.py`_ script for an example. Basically the bootstrap method performs
two steps:

* ``learn("startup.xml")`` reads & parses that file, which contains a single
  pattern "LOAD ALICE", whose action is ``<learn>*.aiml</learn>``
* then, ``respond("load alice")`` executes that loaded pattern, which in turn
  learns all the ``*.aiml`` files

Note: given that ``<learn>*.aiml</learn>`` tries to find all ``*.aiml`` files
in the current directory (since it contains the filename as a base path) the
Python process needs to have the AIML directory *as its current directory* for
it to work. Otherwise it will not find any file and will silently fail.
For that reason, the ``bootstrap()`` method has an optional argument
``chdir`` that makes the it change to that directory before performing any
learn or command execution (but after loadBrain processing). Upon returning
the current directory is moved back to where it was before.





Tests
=====

There are a number of unit tests included (in the ``test`` subdirectory); they 
can be executed by the setup infrastructure as::

  python setup.py test

or they can also be launched by directly calling::

  python test [testname ...]

This last version allows executing only some of the test files by explicitly
naming them in the command line (use the filename without the ``test_`` prefix
and the ``.py`` suffix); if none is specified all will be executed.



.. _PyAIML: https://github.com/cdwfs/pyaiml
.. _Free ALICE AIML set: https://code.google.com/archive/p/aiml-en-us-foundation-alice/
.. _LGPL: http://www.gnu.org/licenses/lgpl.html
.. _ALICE AI Foundation: http://alice.pandorabots.com/
.. _bot.py: aiml/script/bot.py
.. _AIML 1.0.1: http://www.alicebot.org/TR/2011/

------------------------------------------------------------------------------


Original README from PyAIML
===========================


PyAIML is an interpreter for AIML (the Artificial Intelligence Markup
Language), implemented entirely in standard Python.  It strives for
simple, austere, 100% compliance with the AIML 1.0.1 standard, no less
and no more.

This is currently pre-alpha software.  Use at your own risk!

For information on what's new in this version, see the ``CHANGES.txt`` file.

For information on the state of development, including the current level of 
AIML 1.0.1 compliance, see the ``SUPPORTED_TAGS.txt`` file.

Quick & dirty example (assuming you've downloaded the "standard" AIML set)::

	import aiml

	# The Kernel object is the public interface to
	# the AIML interpreter.
	k = aiml.Kernel()

	# Use the 'learn' method to load the contents
	# of an AIML file into the Kernel.
	k.learn("std-startup.xml")

	# Use the 'respond' method to compute the response
	# to a user's input string.  respond() returns
	# the interpreter's response, which in this case
	# we ignore.
	k.respond("load aiml b")

	# Loop forever, reading user input from the command
	# line and printing responses.
	while True: print k.respond(raw_input("> "))


