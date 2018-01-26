..
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.
  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.

Using Facebook Messenger
========================
**Authors**: Jack Urbanek

In ParlAI, you can allow people on Facebook to participate in a ParlAI world as an agent.

Facebookers can be viewed as another type of agent in ParlAI, and hence person-to-person, person-to-bot, or multiple people and bots in group chat can all talk to each other within the same framework.

Facebook agents communicate in observation/action dict format, the same as all other agents in ParlAI. During the conversation, messages are delivered to a person's inbox from the page that you have set up for ParlAI.

.. figure:: _static/img/mturk-small.png
   :align: center

   *Example: TODO put image of a chat window here*

Each messenger task has at least one messenger agent that connects to ParlAI using the Facebook messenger Send/Receive API, encapsulated as a ``MessengerAgent`` object.

Each messenger task also consists of a ``World`` where all agents live and interact within.

Messenger tasks can be grouped together within an ``Overworld`` which can spawn the subtasks and allow people to pick between multiple conversations.

Example Tasks
-------------

We provide two examples of using Facebook Messenger with ParlAI:

- `QA Data Collection <https://github.com/facebookresearch/ParlAI/blob/master/parlai/messenger/tasks/qa_data_collection/>`__: collect questions and answers from people, given a random Wikipedia paragraph from SQuAD.
- `Overworld Demo <https://github.com/facebookresearch/ParlAI/blob/master/parlai/mturk/tasks/overworld_demo/>`__: let people select between three different subtasks, namely an echo bot, a demo of onboarding data collection, and a random chat.

Task 1: Collecting Data
^^^^^^^^^^^^^^^^^^^^^^^

One of the biggest use cases of Messenger in ParlAI is to connect people and collect natural language data.

As an example, the `QA Data Collection task <https://github.com/facebookresearch/ParlAI/blob/master/parlai/messenger/tasks/qa_data_collection/>`__ does the following:

1. Pick a random Wikipedia paragraph from SQuAD dataset.
2. Ask a person to provide a question given the paragraph.
3. Ask the person to provide an answer to their question.

In ``QADataCollectionWorld``, there are two agents: one is the person (``MessengerAgent``), the other is the task agent (``DefaultTeacher`` from SQuAD) that provides the Wikipedia paragraph.

The ``QADataCollectionWorld`` uses ``turn_index`` to denote what stage the conversation is at. One *turn* means that ``world.parley()`` has been called once.

After two turns, the task is finished, and the person's work can be saved during the ``World.shutdown()`` call.


Task 2: Exposing People to Multiple Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Describe this.


Creating Your Own Task
----------------------

**TODO UPDATE ALL OF THIS**

ParlAI provides a generic MTurk dialog interface that one can use to implement any kind of dialog tasks. To create your own task, start with reading the tutorials on the provided examples, and then copy and modify the example ``worlds.py``, ``run.py`` and ``task_config.py`` files to create your task.

A few things to keep in mind:

1. To end a conversation, you should send a message with ``episode_done = True`` from the first non-MTurk agent, and the conversation is ended after all MTurk agents respond.
2. In ``run.py``, You can use ``hit_index`` and ``assignment_index`` to differentiate between different HITs and assignments, and change the content of the task accordingly.
3. Make sure to test your dialog task using MTurk's sandbox mode before pushing it live, by using the ``--sandbox`` flag (enabled by default) when running ``run.py``.
4. [Optional] If you want to show a custom webpage (instead of the default one) for any of your MTurk agents, you can create an ``html`` folder within your task directory, and then create the ``<mturk_agent_id>_cover_page.html`` and ``<mturk_agent_id>_index.html`` files within the ``html`` directory. In those files, you can extend from ``core.html`` and override any code blocks that you want to change. (Please look at `parlai/mturk/core/html/mturk_index.html <https://github.com/facebookresearch/ParlAI/blob/master/parlai/mturk/core/html/mturk_index.html>`__ as an example.) These agent-specific templates will automatically be shown to the Turkers in the next run.


Running a Task
--------------

**TODO UPDATE ALL OF THIS**

If you have not used Mechanical Turk before, you will need an MTurk Requester Account and an AWS account (these are two separate accounts). Follow the steps below:

- Sign up for an AWS account at `aws.amazon.com <https://aws.amazon.com/>`__

- Sign up for an MTurk account at `requester.mturk.com <https://requester.mturk.com/>`__

- Go to the developer tab (`https://requester.mturk.com/developer <https://requester.mturk.com/developer>`__) and link your AWS account to your MTurk account (Step 2 on that screen)

- MTurk also has a “Sandbox” which is a test version of the MTurk marketplace. You can use it to test publishing and completing tasks without paying any money. ParlAI supports the Sandbox. To use the Sandbox, you need to sign up for a `Sandbox account <http://requestersandbox.mturk.com/>`__. You will then also need to `link your AWS account <http://requestersandbox.mturk.com/developer>`__ to your Sandbox account. In order to test faster, you will also want to create a `Sandbox Worker account <http://workersandbox.mturk.com/>`__. You can then view tasks your publish from ParlAI and complete them yourself.

- ParlAI's MTurk functionality requires a free heroku account which can be obtained `here <https://signup.heroku.com/>`__. Running any ParlAI MTurk operation will walk you through linking the two.

Then, to run an MTurk task, first ensure that the task directory is in `parlai/mturk/tasks/ <https://github.com/facebookresearch/ParlAI/blob/master/parlai/mturk/tasks/>`__. Then, run its ``run.py`` file with proper flags:

.. code-block:: console

    python run.py -nc <num_conversations> -r <reward> [--sandbox]/[--live]

E.g. to create 2 conversations for the `QA Data Collection <https://github.com/facebookresearch/ParlAI/blob/master/parlai/mturk/tasks/qa_data_collection/>`__ example with a reward of $0.05 per assignment in sandbox mode, first go into the task directory and then run:

.. code-block:: console

    python run.py -nc 2 -r 0.05 --sandbox

Please make sure to test your task in MTurk sandbox mode first (``--sandbox``) before pushing it live (``--live``).

Additional flags can be used for more specific purposes.

- ``--unique`` ensures that an Turker is only able to complete one assignment, thus ensuring each assignment is completed by a unique person.

- ``--allowed-conversations <num>`` prevents a Turker from entering more than <num> conversations at once (by using multiple windows/tabs). This defaults to 0, which is unlimited.

- ``--count-complete`` only counts completed assignments towards the num_conversations requested. This may lead to more conversations being had than requested (and thus higher costs for instances where one Turker disconnects and we pay the other) but it ensures that if you request 1,000 conversations you end up with at least 1,000 completed data points.
