Data Handling and Batching
==========================
**Authors**: Alexander Holden Miller, Kurt Shuster

.. note::
    If you are unfamiliar with the basics of displaying data or
    calling train or evaluate on a model, please first see
    the `getting started <tutorial_basic.html>`_ section.
    If you are interested in creating a task, please see
    `that section <tutorial_task.html>`_.

Introduction
^^^^^^^^^^^^

This tutorial will cover the details of batched data, and why we use Shared Worlds.

With relatively small modifications to a basic agent, it will be able to support
multithreading and batching. If you need extra speed or are using a very large
dataset which does not fit in memory, we can use a multiprocessed pytorch
dataloader for improved performance.

First, let's consider a diagram of the basic flow of DialogPartnerWorld,
a simple world with two conversing agents.

.. image:: _static/img/world_basic.png

The teacher generates a message, which is shown to the agent.
The agent generates a reply, which is seen by the teacher.


Expanding to batching using share()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For all tasks one might make,
there's one function we need to support for batching: ``share()``.
This function should provide whatever is needed to set up a "copy" of the original
instance of the agent for either each row of a batch.

We create shared agents by instantiating them in the following way:

.. code-block:: python

    Agent0 = Agent(opt)
    ...
    Agent1 = Agent(opt, Agent0.share())
    Agent2 = Agent(opt, Agent0.share())
    Agent3 = Agent(opt, Agent0.share())

.. image:: _static/img/world_share.png

That is, the executed are:

1. We set up a starting instance of the world: that is, we use ``create_task``
   to set up a base world with the appropriate agents and tasks.
2. We pass this world to a ``BatchWorld``.
3. We create ``batchsize`` worlds, each initialized from a ``share()``'d
   version of the world and the agents therein.

Now, every time we call ``parley`` on this BatchWorld, we will complete
``batchsize`` examples.

.. image::  _static/img/world_batchbasic.png

.. note::
    So far, our diagram is exactly the same as Hogwild. We'll see how it can
    change below when agents implement the ``batch_act`` function
    (as GPU-based models will).


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
      instance of the agent (not the copies) has a function named ``batch_act``
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

.. image::  _static/img/world_batchagent.png

Now, the agent can process the entire batch at once. This is especially helpful
for GPU-based models, which prefer to process more examples at a time.

Tip: if you implement ``batch_act()``, your ``act()`` method can just call ``batchact()``
and pass the observation it is supposed to process in a list of length 1.

Of course, this also means that we can use batch_act in both the task and the
agent code:

.. image::  _static/img/world_batchboth.png
