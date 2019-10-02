..
  Copyright (c) Facebook, Inc. and its affiliates.
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.

Creating a Torch Agent
======================
**Author**: Stephen Roller, Alexander Holden Miller

In this tutorial, we'll be setting up an agent which learns from the data it
sees to produce the right answers.

For this agent, we'll be implementing a simple GRU Seq2Seq agent based on
Sequence to Sequence Learning with Neural Networks (Sutskever et al. 2014) and
Sean Robertson's `Seq2Seq PyTorch tutorial
<http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`_.


Naming Things
^^^^^^^^^^^^^
In order to make programmatic importing easier, we use a simple naming scheme
for our models, so that on the command line we can just type "--model seq2seq"
("-m seq2seq") to load up the seq2seq model.

To this end, we create a folder under parlai/agents with the name seqseq, and
then put an empty ``__init__.py`` file there along with seq2seq.py.
Then, we name our agent "Seq2seqAgent".

The ParlAI argparser automatically tries to translate "--model seq2seq" to
"parlai.agents.seq2seq.seq2seq:Seq2seqAgent".
Underscores in the name become capitals in the class name: "--model local_human"
resides at "parlai.agents.local_human.local_human:LocalHumanAgent".

If you need to put a model at a different path, you can specify the full path
on the command line in the format above (with a colon in front of the class name).
For example, "--model parlai.agents.remote_agent.remote_agent:ParsedRemoteAgent".


Annotated Implementation
^^^^^^^^^^^^^^^^^^^^^^^^

First we'll bring in minimal imports and define the actual encoder and decoder.
If you find this code to be obtuse, we recommend you begin with the
`Learning PyTorch with Examples
<https://pytorch.org/tutorials/beginner/pytorch_with_examples.html>` tutorial.

Full Implementation & running this model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can see the full code for this `here
<https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/example_seq2seq/example_seq2seq.py>`_.

You can try this model now with a command like the following:

.. code-block:: bash

    # batchsize 32
    python examples/train_model.py -t babi:task10k:1 --dict-file /tmp/dict_babi:task10k:1 -bs 32 -vtim 30 -m example_seq2seq

    # using the adam optimizer
    python examples/train_model.py -t babi:task10k:1 --dict-file /tmp/dict_babi:task10k:1 -bs 32 -vtim 30 -m example_seq2seq -opt adam -lr 3e-4
