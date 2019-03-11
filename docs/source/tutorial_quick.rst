..
  Copyright (c) Facebook, Inc. and its affiliates.
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.

ParlAI Quick-start
==================
**Authors**: Alexander Holden Miller


Install
-------

1. Clone ParlAI Repository:

.. code-block:: bash

    git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI

2. Install ParlAI:

.. code-block:: bash

    cd ~/ParlAI; python setup.py develop

3. Several models have additional requirements, such as `PyTorch <http://pytorch.org/>`_.


View a task & train a model
---------------------------

Let's start by printing out the first few examples of the bAbI tasks, task 1.

.. code-block:: bash

  # display examples from bAbI 10k task 1
  python examples/display_data.py -t babi:task10k:1

Now let's try to train a model on it (even on your laptop, this should train fast).

.. code-block:: bash

  # train MemNN using batch size 8 and 4 threads for 5 epochs
  python examples/train_model.py -t babi:task10k:1 -mf /tmp/babi_memnn -bs 8 -nt 4 -eps 5 -m memnn --no-cuda

Let's print some of its predictions to make sure it's working.

.. code-block:: bash

  # display predictions for model save at specified file on bAbI task 1
  python examples/display_model.py -t babi:task10k:1 -mf /tmp/babi_memnn -ecands vocab

The "eval_labels" and "MemNN" lines should (usually) match!

Let's try asking the model a question ourselves.

.. code-block:: bash

  # interact with saved model
  python examples/interactive.py -mf /tmp/babi_memnn
  ...
  Enter your message: John went to the hallway.\n Where is John?

Hopefully the model gets this right!


Simple model
------------

Let's put together a super simple model which will print the parsed version of what is said to it.

First let's set it up:

.. code-block:: bash

  mkdir parlai/agents/parse
  touch parlai/agents/parse/parse.py

We'll inherit the TorchAgent parsing code so we don't have to write it ourselves.

.. code-block:: python

  from parlai.core.torch_agent import TorchAgent, Output

  class ParseAgent(TorchAgent):
      def eval_step(self, batch):
          # for each row in batch, convert tensor to string
          return Output([str(row) for row in batch.text_vec])

Now let's test it out:

.. code-block:: bash

  python examples/display_model.py -t babi:task10k:1 -m parse

You'll notice the model is always outputting the index for the "unknown" token.
This token is automatically selected because the dictionary doesn't recognize any tokens,
because we haven't built a dictionary yet. Let's do that now.

.. code-block:: bash

  python examples/build_dict.py -t babi:task10k:1 -df /tmp/parse.dict

Now let's try our parse agent again.

.. code-block:: bash

  python examples/display_model.py -t babi:task10k:1 -m parse -df /tmp/parse.dict

The ParseAgent overrides one of two abstract functions in TorchAgent: ``train_step`` and ``eval_step``.
Overriding these functions allow you to build an agent quickly by implementing just the most
typical custom code for a model, and inheriting vectorization and batching from TorchAgent.

You can override any functions to change the default argument values or to override the behavior with your own.
For example, you could change the vectorizer to return numpy arrays instead of Torch Tensors.


Conclusion
----------

To see more details about ParlAI's general structure, how tasks and models are set up,
or how to use Mechanical Turk, Messenger, Tensorboard, and more--check out the other tutorials.
