..
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.
  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.

Creating an Agent
=================

In this tutorial, we'll be setting up an agent which learns from the data it
sees to produce the right answers.

For this agent, we'll be implementing a simple GRU Seq2Seq agent based on
Sequence to Sequence Learning with Neural Networks (Sutskever et al. 2014) and
Sean Robertson's `Seq2Seq PyTorch tutorial <http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`_.

Part 1: Naming Things
^^^^^^^^^^^^^^^^^^^^^^^^^
    *"There are two hard problems in computer science:
    cache invalidation, naming things, and off-by-one errors."*

In order to make programmatic importing easier, we use a simple naming scheme
for our models, so that on the command line we can just type "--model seq2seq"
to load up the seq2seq model.

To this end, we create a folder under parlai/agents with the name seqseq, and
then put an empty __init__.py file there along with seq2seq.py.
Then, we name our agent "Seq2seqAgent".

This way, "--model seq2seq" can translate to "parlai.agents.seq2seq.seq2seq:Seq2seqAgent".
Underscores in the name become capitals in the class name: "--model local_human"
resides at "parlai.agents.local_human.local_human:LocalHumanAgent".
If you need to put a model at a different path, you can specify the full path
on the command line in the format above (with a colon in front of the class name).
For example, "--model parlai.agents.remote_agent.remote_agent:ParsedRemoteAgent".

Part 2: Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^
First off, generally we should inherit from the Agent class in parlai.core.agents.
This provides us with some default implementations (often, ``pass``) of some utility
functions like "shutdown".

First let's focus on the primary functions to implement, then we can come back and add more functionality.

.. code-block:: python
    def __init__(self, opt, shared=None):
