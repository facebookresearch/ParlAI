..
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.
  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.

Batching and Hogwild
====================
**Author**: Alexander Holden Miller

Summary
^^^^^^^
There's one function we need to support for both hogwild and batching: ``share()``.

This function should provide whatever is needed to set up a "copy" of the original
instance of the agent for either each row of a batch or each thread in hogwild.

The same function is used for both batching and hogwild, since most agents only
use one or the other. However, an agent may check the ``numthreads`` and/or
``batchsize`` options to adjust its behavior if it wants to support both, and
we do support doing both batching and hogwild at the same time if the agent
desires.


Hogwild (multiprocessing)
^^^^^^^^^^^^^^^^^^^^^^^^^
Hogwild is initialized in the following way:

1. We set up a starting instance of the world: that is, we use ``create_task``
   to set up a base world with the appropriate agents and tasks.
2. We pass this world to a ``HogwildWorld``, which sets up a number of
   synchronization primitives
3. We launch ``numthreads`` threads, each initialized from a ``share()``'d
   version of the world and the agents therein.
4. Once these threads and their world copies are all launched, we return control back

Now that this world is set up, every time we call parley on it, it will release
one of its threads to do a parley with its copy of the original base world.

There's some added complexity in the implementation of the class to manage
synchronization primitives, but the Hogwild world should generally behave just
like a regular World, so you shouldn't need to worry about it. If you do want
to check out the implementation, look for HogwildWorld in the `core/worlds.py file
<https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/worlds.py>`_.

Sharing needs to be implemented properly within all these agents and worlds so
all the proper information is shared and synced between the threads. We'll take
a look at the common setup needs for each of those.


Hogwild Teachers
~~~~~~~~~~~~~~~~
The default setup for teachers include creating a Metrics object to track
different metrics, including the number of examples shown, accuracy, and f1.
The default ``share()`` method automatically sets up a thread-safe version of
these metrics if needed--children can go ahead and access these metrics as normal.

Teachers using dynamic data can most likely proceed as normal, without syncing
any information outside of the metrics class. However, fixed datasets need
mechanisms built in to make sure that they don't do validation or testing
examples more or less than once to ensure consistent results.

Fortunately, the FixedDialogTeacher has this all built in already,
so merely extending that class provides all the needed functionality.


Hogwild Models
~~~~~~~~~~~~~~
For models using hogwild training, the primary concern is to share a thread-safe
version of the model. This process will vary based on which framework you're
using, but we'll include a few tips for PyTorch here.

First, check out the best practices here:
http://pytorch.org/docs/master/notes/multiprocessing.html

The primary things to remember are
1. call ``model.share_memory()`` and include your model in the ``share()`` function
2. make sure to switch the multiprocessing start method if CUDA is enabled

You can see an example of this in the `Starspace model
<https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/starspace/starspace.py>`_.
This model uses multiple CPU threads for faster training, and does not use GPUs at all.

Showing only the code relevant to model sharing, we see:

.. code-block:: python

    def __init__(self, opt, shared=None):
        if shared:
            torch.set_num_threads(1)  # otherwise torch uses multiple cores for computation
            self.model = shared['model']  # don't set up model again yourself
        else:
            self.model = Starspace(opt, len(self.dict), self.dict)
            self.model.share_memory()

    def share(self):
        shared = super().share()
        shared['model'] = self.model
        return shared


Batching
^^^^^^^^
Batching is set up in the following way (the first step is the same as Hogwild):

1. We set up a starting instance of the world: that is, we use ``create_task``
   to set up a base world with the appropriate agents and tasks.
2. We pass this world to a ``BatchWorld``.
3. We create ``batchsize`` worlds, each initialized from a ``share()``'d
   version of the world and the agents therein.

Now, every time we call ``parley`` on this BatchWorld, we will complete ``batchsize`` examples.
Note that this is different than the behavior of HogwildWorld, where only a
single example is executed for each call to parley.

There's a few more complex steps to actually completing a parley in this world.

1. Call ``parley_init`` on each shared world, if the world has it implemented.
   Most classes don't need this--we currently only use it for our ``MultiWorld``,
   which handles the case when one specifies multiple separate tasks to launch
   (e.g. "-t babi,squad"). This does any pre-parley setup, here choosing which
   sub-tasks to use in the next parley.
2. Then, iterate over the number of agents involved in the task. For most tasks,
   this is just two agents: the teacher (task) and the student (model). For each
   agent, we do two steps:

   a. Call ``BatchWorld.batch_act()``. This method first checks if the **original**
      isntance of the agent (not the copies) has a function named ``batch_act``
      implemented and does not have an attribute ``use_batch_act`` set to ``False``.
      This function is described more below. If condition is not met,
      the BatchWorld's ``batch_act`` method iterates through each agent copy in the
      batch and calls the ``act()`` method for that instance.
      This is the default behavior in most circumstances, and allows agents to
      immediately work for batching without any extra work--the batch_act method
      is merely for improved efficiency.
   b. Call ``BatchWorld.batch_observe()``. This method goes through every **other**
      agent, and tries to call the ``observe()`` method on those agents. This gives
      other agents (usually, just the one other agent) the chance to see the action
      of the agent whose turn it is to act currently.

Next, we'll look at how teachers and models can take advantage of the setup
above to improve performance.


Batched Teachers
~~~~~~~~~~~~~~~~
Batched teachers need to consider everything that a Hogwild Teacher does (see above)
except for thread safety--for example, they also need to make sure they sync
which example index they are on so that they don't repeat or skip valid/test examples.

However, teachers can do some tricks for batching which can help training by
reducing the amount of zero-padding in tensors constructed from batches of text.
This technique alone sped up the time to train on a single epoch of WMT De-En
with a simple convolutional architecture by approximately 4x.
See `this paper <https://arxiv.org/abs/1706.05765>`__ for an analysis of the
impact of different strategies on speed and convergence.

As before, the FixedDialogTeacher handles all of this for you.

In order to reduce the zero-padding in examples, the FixedDialogTeacher first
processes the entire base dataset, squashing episodes into a single example
if there are multiple examples in each episode. For every example
in an episode, a separate squashed episode will be created from the examples up
to and including the current example.

The squashing can be controlled by two command-line arguments, which set
whether the labels are included in the squashing (you want them for dialog,
but you might not for question-answering),
as well as how many examples from the past should be included.

:param batch-sort: (bool, default True) whether to do squashing & batch sorting at all
:param include-labels: (bool, default True) whether to include labels in the context.
:param context-length: (int, default -1) how many past examples in the episode to
                       include in the context for the current one. default -1 is all.
                       note that some datasets have **very** long episodes (e.g.
                       OpenSubtitles has episodes with hundreds of examples), so
                       setting context-length to a smaller value can limit the
                       context to an approachable amount of information for the model
                       as well as limiting memory usage.

After doing the squashing, the new episodes are sorted by number of spaces
(which roughly translates into the number of tokens after parsing the text)
and then split into batches where each batch contains examples of similar size.

The FixedDialogTeacher then implements a ``batch_act()`` method to return batches
from these batches instead of using the regular ``act()`` method to return single examples.

Let's look at a quick example to make sure the flattening is clear.
Consider the following "conversation", where the ``x``'s represent 'text' fields
and the ``y``'s represent labels in a continuous conversation between two agents.

.. code-block:: python

  x1 y1
  x2 y2
  x3 y3
  x4 y4

Without batching, these examples will be presented to the agent over four parleys:

.. code-block:: python

    {'text': x1, 'labels': [y1], 'episode_done': False}
    {'text': x2, 'labels': [y2], 'episode_done': False}
    {'text': x3, 'labels': [y3], 'episode_done': False}
    {'text': x4, 'labels': [y4], 'episode_done': True}

Using the flattening strategy above, with ``context-length`` set to -1 and
``include-labels`` set to False (not recommended for conversations),
in separate rows of a batch we'd get:

.. code-block:: python

    {'text': x1,                'labels': [y1], 'episode_done': True}
    {'text': x1 + x2,           'labels': [y2], 'episode_done': True}
    {'text': x1 + x2 + x3,      'labels': [y3], 'episode_done': True}
    {'text': x1 + x2 + x3 + x4, 'labels': [y4], 'episode_done': True}

If we change ``context-length`` to 3 and ``include-labels`` to True:

.. code-block:: python

    {'text': x1,           'labels': [y1], 'episode_done': True}
    {'text': x1 + y1 + x2, 'labels': [y2], 'episode_done': True}
    {'text': x2 + y2 + x3, 'labels': [y3], 'episode_done': True}
    {'text': x3 + y3 + x4, 'labels': [y4], 'episode_done': True}


Batched Models
~~~~~~~~~~~~~~
Finally, models need to be able to handle observations arriving in batches.

The first key concept to remember is that, if the model agent implements
``batch_act()``, **act will not be called** as long as ``batchsize`` > 1.

However, copies of the agent will still be created, and the ``observe`` method
of each one will be called. This allows each copy to maintain a state related
to a single row in the batch. Remember, since each row in the batch is represented
by a separate world, they are completely unrelated. This means that the model
only needs to be set up in the original instance, and need not be shared with
its copies.

The ``observe()`` method returns a (possibly modified) version of the observation
it sees, which are collected into a list for the agent's ``batch_act()`` method.

Now, the agent can process the entire batch at once. This is especially helpful
for GPU-based models, which prefer to process more examples at a time.

Tip: if you implement ``batch_act()``, your ``act()`` method can just call ``batchact()``
and pass the observation it is supposed to process in a list of length 1.
