..
  Copyright (c) Facebook, Inc. and its affiliates.
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.

What is ParlAI?
===============
**Authors**: Alexander Holden Miller, Jason Weston

It's a python-based platform for enabling dialog AI research.

Its goal is to provide researchers:

- a unified framework for sharing, training and testing dialog models
- many popular datasets available all in one place, with the ability to multi-task over them
- seamless integration of :doc:`Amazon Mechanical Turk <tutorial_mturk>` for data collection and human evaluation
- integration with :doc:`Facebook Messenger <tutorial_messenger>` to connect agents with humans in a chat interface

You can also see the `README <https://github.com/facebookresearch/ParlAI/blob/master/README.md>`_ for more basic info on ParlAI, or continue reading this document.


Getting Started
---------------

Observations
^^^^^^^^^^^^
Python dictionaries containing different types of information are the primary
way messages are passed between agents and the environment in ParlAI.

The :doc:`observations <observations>` documentation goes into more detail about
each field, but the following table shows the basics.


.. image:: _static/img/act-obs-dict.png
    :width: 60 %

All of these fields are technically optional, and each task should use them
according to what kind of information is available in that task (for example,
not all tasks contain explicit rewards, or a set of candidate labels to choose from).

Dataset-specific fields are available in some cases in order to support
reproducing paper results. For example, SQuAD has an ``answer_starts`` field,
which is available in the "squad:index" task.

**Note**: during validation and testing, the ``labels`` field is renamed
``eval_labels``--this way, the model won't accidentally train on the labels,
but they are still available for calculating model-side loss.
Models can check if they are training on a supervised task in the following manner:

.. code-block:: python

    is_training = 'labels' in observation


Agents
^^^^^^

The most basic concept in ParlAI is an Agent.
Agents can be humans, a simple bot which repeats back anything that it hears,
your perfectly-tuned neural network, a dataset being read out,
or anything else that might send messages or interact with its environment.

Agents have two primary methods they need to define:

.. code-block:: python

    def observe(self, observation): # update internal state with observation
    def act(self): # produce action based on internal state

``observe()`` notifies the agent of an action taken by another agent.

``act()`` produces an action from the agent.


Teachers
^^^^^^^^

A Teacher is special type of agent. They also implement the ``act`` and ``observe``
functions like any agent does, but they also keep track of metrics which they
return via a ``report`` function, such as the number of questions they have posed
or how many times those questions have been answered correctly.

Datasets typically implement a subclass of Teacher, providing functions which
download the dataset from its source if necessary, read the file into the
right format, and provide an example with each call to the teacher's ``act``
function.

Exchanges between a student Agent and a bAbI task Teacher might look like the following dicts:

.. code-block:: python

    Teacher: {
        'text': 'Sam went to the kitchen\nPat gave Sam the milk\nWhere is the milk?',
        'labels': ['kitchen'],
        'label_candidates': ['hallway', 'kitchen', 'bathroom'],
        'episode_done': False  # indicates next example will be related to this one
    }
    Student: {
        'text': 'hallway'
    }
    Teacher: {
        'text': 'Sam went to the hallway\nPat went to the bathroom\nWhere is the milk?',
        'labels': ['hallway'],
        'label_candidates': ['hallway', 'kitchen', 'bathroom'],
        'episode_done': True
    }
    Student: {
        'text': 'hallway'
    }
    Teacher: {
        ... # starts next episode
    }
    ...

Worlds
^^^^^^

Worlds define the environment in which agents interact with one another. Worlds
must implement a ``parley`` method, which conducts one set of interactions with
each call.

A simple world included in ParlAI, which all of our currently included tasks use,
is the ``DialogPartnerWorld``. DialogPartnerWorld is initialized with two agents,
and with each call to ``parley``, one exchange is done between the agents, in
the following manner:

.. code-block:: python

    query = teacher.act()
    student.observe(query)
    reply = student.act()
    teacher.observe(reply)

Another simple world we include is MultiAgentDialogWorld, which is similar
but generalizes this to cycle between any number of agents in a round robin
fashion.

Advanced Worlds
^^^^^^^^^^^^^^^

We also include a few more advanced "container" worlds: in particular, we include both a
BatchWorld and a HogwildWorld. These worlds are automatically used when either
the ``numthreads`` parameter or the ``batchsize`` parameter are set to greater
than one. Some extra functionality is needed to get these to work on the side
of both the teacher and the learner, but we'll cover that in a different
tutorial (see: :doc:`tutorial_worlds`).

Simple Display Data Loop
^^^^^^^^^^^^^^^^^^^^^^^^

Now that we understand the basics, let's set up a simple loop which displays
whichever task we specify. A complete version of this for utility is included
in the ``examples`` directory (in ``display_data.py``), but we'll do this one from scratch.

First, a few imports:

.. code-block:: python

    from parlai.core.agents import Agent
    from parlai.core.params import ParlaiParser
    from parlai.core.worlds import create_task

The Agent class will be the parent class for our own agent, which we'll implement here.
The ``ParlaiParser`` provides a set of default command-line arguments and
parsing, and create_task allows us to automatically set up the right world and
teacher for a named task from the set of tasks available within ParlAI.

First, we'll define our agent, which just repeats back the correct answer if
available or else says "I don't know."


.. code-block:: python

    class RepeatLabelAgent(Agent):
        # initialize by setting id
        def __init__(self, opt):
            self.id = 'LabelAgent'
        # store observation for later, return it unmodified
        def observe(self, observation):
            self.observation = observation
            return observation
        # return label from before if available
        def act(self):
            reply = {'id': self.id}
            if 'labels' in self.observation:
                reply['text'] = ', '.join(self.observation['labels'])
            else:
                reply['text'] = "I don't know."
            return reply


Now that we have our our agent, we'll set up the display loop.

.. code-block:: python

    parser = ParlaiParser()
    opt = parser.parse_args()

    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    for _ in range(10):
        world.parley()
        print(world.display())
        if world.epoch_done():
            print('EPOCH DONE')
            break

And that's it! The world.display() automatically cycles through each of the
world's agents and displays their last action.  NOTE, if you want to get at and
look at the data from here rather than calling
world.display() you could access world.acts directly:

.. code-block:: python

    parser = ParlaiParser()
    opt = parser.parse_args()

    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    for _ in range(10):
        world.parley()
	for a in world.acts:
	    # print the actions from each agent
	    print(a)
        if world.epoch_done():
            print('EPOCH DONE')
            break


If you run this on the command
line, you can specify which task to show by setting '-t {task}'.

Tasks are specified in the following format:

* '-t babi' sets up the ``DefaultTeacher`` in 'parlai/core/tasks/babi/agents.py'.

* '-t babi:task1k' sets up the ``Task1kTeacher`` in the babi/agents.py file, which allows
  you to specify specific settings for certain tasks. For bAbI, this refers to the setting
  where there are only 1000 unique training examples per task.

* '-t babi:task1k:1' provides 1 as a parameter to ``Task1kTeacher``, which is interpreted
  by the Task1kTeacher to mean "I want task 1" (as opposed to the 19 other bAbI tasks).

* '-t babi,squad' sets up the ``DefaultTeacher`` for both babi and squad. Any number
  of tasks can be chained together with commas to load up each one of them.

* '-t #qa' specifies the 'qa' category, loading up all tasks with that category
  in the 'parlai/core/task_list.py' file.


These flags are used across ParlAI, here are some examples of using them for
displaying data with the existing script
`display_data <https://github.com/facebookresearch/ParlAI/blob/master/parlai/scripts/display_data.py>`_:

.. code-block:: python

   #Display 10 random examples from task 1 of the "1k training examples" bAbI task:
   python examples/display_data.py -t babi:task1k:1

   #Displays 100 random examples from multi-tasking on the bAbI task and the SQuAD dataset at the same time:
   python examples/display_data.py -t babi:task1k:1,squad -n 100


The `--task` flag (`-t`  for short) specifies the task and `--datatype` (`-dt`) specifies
train, valid or test. Note that `train:stream` or `valid:stream` can be specified
to denote that you want the data to stream online if possible, rather than loading into memory,
and `train:ordered` can be specified, otherwise data from the train set comes in a random order by
default (whereas valid and test data is ordered by default).


Validation and Testing
^^^^^^^^^^^^^^^^^^^^^^

During validation and testing, the 'labels' field is removed from the observation dict.
This tells the agent not to use these labels for training--however, the labels are
still available via the 'eval_labels' field in case you need to compute model-side
metrics such as perplexity.
These modes can be set from the command line with '-dt valid' / '-dt test'.
You can also set 'train:evalmode' if you want to look at the train data in the same way
as the test data (with labels hidden).

Now, our RepeatLabel agent no longer has anything to say. For datasets which provide a set
of candidates to choose from ('label_candidates' in the observation dict), we
can give our agent a chance of getting the answer correct by replying with one
of those.

Let's modify our agent's act function to select a random label candidate when
the labels aren't available:

.. code-block:: python

    import random

    def act(self):
        reply = {'id': self.id}
        if 'labels' in self.observation:
            reply['text'] = ', '.join(self.observation['labels'])
        elif 'label_candidates' in self.observation:
            cands = self.observation['label_candidates']
            reply['text'] = random.choice(list(cands))
        else:
            reply['text'] = "I don't know."
        return reply


Of course, we can do much better than randomly guessing. In another tutorial,
we'll set up a better agent which learns from the training data.


Training and Evaluating Existing Agents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For now, we'll look at the main calls for evaluating and
training an agent that is already coded.
We can use the scripts
`train_model <https://github.com/facebookresearch/ParlAI/blob/master/parlai/scripts/train_model.py>`_
and `eval_model <https://github.com/facebookresearch/ParlAI/blob/master/parlai/scripts/eval_model.py>`_.
Here are some examples:

.. code-block:: python

   #Evaluate on the bAbI test set with a human agent (using the local keyboard as input):
   python examples/eval_model.py -m local_human -t babi:Task1k:1 -dt valid

   #Evaluate an IR baseline model on the validation set of the Movies Subreddit dataset:
   python examples/eval_model.py -m ir_baseline -t "#moviedd-reddit" -dt valid

   #Display the predictions of that same IR baseline model:
   python examples/display_model.py -m ir_baseline -t "#moviedd-reddit" -dt valid

   #Train a seq2seq model on the "10k training examples" bAbI task 1 with batch size of 32 examples until accuracy reaches 95% on validation (requires pytorch):
   python examples/train_model.py -t babi:task10k:1 -m seq2seq -mf /tmp/model_s2s -bs 32 -vtim 30 -vcut 0.95


   #Trains an attentive LSTM model on the SQuAD dataset with a batch size of 32 examples (pytorch and regex):
   python examples/train_model.py -m drqa -t squad -bs 32 -mf /tmp/model_drqa

   #Tests an existing attentive LSTM model (DrQA reader) on the SQuAD dataset from our model zoo:
   python examples/eval_model.py -t squad -mf "models:drqa/squad/model"


The main flags are:

1) `-m` (`-model`) which sets the agent type that will be trained.
The agents available in parlAI `are here <https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents>`_.
See `this tutorial <tutorial_task.html>`_ for making your own agents.

2) `-mf` (`--modelfile`) points to the file name of where to save your model.

3) `-t` (`--task`) as described before.

Of course every model has various parameters and hyperparameters to set in general.


**Model Zoo**

A new feature in ParlAI is that it also now maintains a *model zoo* of existing model files of agents that have been trained on tasks. See the devoted documentation section or `here for details <https://github.com/facebookresearch/ParlAI/blob/master/parlai/zoo/model_list.py>`_.

The set of agents and models in the model zoo in ParlAI is continually growing from contributors.


Tasks
^^^^^

The set of tasks in ParlAI can be found in the task list in the `code here <https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/task_list.py>`_ or in this `documentation
here <tasks.html>`_. See `this tutorial <tutorial_task.html>`_ for making your own tasks.

ParlAI downloads the data required for a requested task automatically (using the build.py code in the task)
and will put it in your `--datapath`, which is configurable, but by default will be in
ParlAI/data (but you can point this e.g. to another disk with more memory).
It only downloads the tasks you request.

The set of tasks in ParlAI is continually growing from contributors.
