Running crowdsourcing tasks
=====================
__Authors__: Jack Urbanek, Emily Dinan, Will Feng, Eric Smith

:::{warning}
ParlAI's MTurk functionality has expanded out of this project to become [Mephisto](https://github.com/facebookresearch/Mephisto), and we have moved our crowdsourcing code from `parlai.mturk` into [`parlai.crowdsourcing`](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing). See [this README](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/README.md) for more specifics on how to run crowdsourcing tasks in the current version of ParlAI.

If you wish to access the old version of this tutorial for pre-Mephisto crowdsourcing tasks, switch to the `final_mturk` tag of ParlAI:
```bash
git checkout final_mturk
```
:::

In ParlAI, you can use the crowdsourcing platform [Amazon Mechanical Turk](https://www.mturk.com/) for __data collection__,
__training__, or __evaluation__ of your dialog model.

Human Turkers are viewed as just another type of agent in ParlAI; hence,
agents in a group chat consisting of any number of humans and/or bots
can communicate with each other within the same framework.

The human Turkers communicate in an observation/action dict format, the
same as all other agents in ParlAI. During the conversation, human
Turkers receive a message that is rendered on the live chat webpage,
such as the following:

![*Example: Human Turker participating in a QA data collection
task*](_static/img/mturk-small.png)

Each crowdsourcing task has at least one human Turker who connects to ParlAI
via the Mechanical Turk Live Chat interface, encapsulated as an
`Agent` object.

Each crowdsourcing task also consists of a `World` in which all agents live and
interact.

Example Tasks
-------------

We provide a few examples of using crowdsourcing tasks with ParlAI:

- [Chat demo](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/tasks/chat_demo): have two humans chat back and forth for a multi-turn conversation.
- [Model chat](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/tasks/model_chat): have a human chat with a model agent in a conversation, perhaps about an image, and optionally have the human select among checkboxes to annotate the model's responses.
- [Static turn annotations](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/tasks/turn_annotations_static): have a human read a static conversation between two partners and select among checkboxes to annotate one of the speakers' responses.
-   [QA data
    collection](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/tasks/qa_data_collection/):
    collect questions and answers from Turkers, given a random Wikipedia
    paragraph from SQuAD.
- [ACUTE-Eval](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/tasks/acute_eval): run a comparison test where a human reads two conversations and chooses one or the other based on an evaluation questions such as, "Who would you prefer to talk to for a long conversation?""

### Task 1: Collecting Data

One of the biggest use cases of Mechanical Turk is to collect natural
language data from human Turkers.

As an example, the [QA data collection
task](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/tasks/qa_data_collection/)
does the following:

1.  Pick a random Wikipedia paragraph from the SQuAD dataset.
2.  Ask a Turker to provide a question given the paragraph.
3.  Ask the same Turker to provide an answer to their question.

### {{{TODO: revise this section onward}}}

In `QADataCollectionWorld`, there are two agents: one is the human
Turker (`MTurkAgent`), the other is the task agent (`DefaultTeacher`
from SQuAD) that provides the Wikipedia paragraph.

The `QADataCollectionWorld` uses `turn_index` to denote what stage the
conversation is at. One *turn* means that `world.parley()` has been
called once.

After two turns, the task is finished, and the Turker's work is
submitted for review.

### Task 2: Evaluating a Dialog Model

You can easily evaluate your dialog model's performance with human
Turkers using ParlAI. As an example, the [Model Evaluator
task](https://github.com/facebookresearch/ParlAI/blob/master/parlai/mturk/tasks/model_evaluator/)
does the following:

1.  Initialize a task world with a dialog model agent
    ([ir\_baseline](https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/ir_baseline/ir_baseline.py#L98))
    and a dataset
    ([MovieDD-Reddit](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/moviedialog/agents.py#L57)).
2.  Let all the agents in the task world `observe()` and `act()` once,
    by calling `parley()` on the world.
3.  Ask the human Turker to rate the dialog model agent's response on a
    scale of 0-10.

In `ModelEvaluatorWorld`, there are two main components: one is the
`task_world` that contains the task and the dialog model we are
evaluating, the other is the `MTurkAgent` which is an interface to the
human Turker.

Note that since the human Turker speaks only once to provide the rating,
the `ModelEvaluatorWorld` doesn't need to use `turn_index` to keep track
of the turns.

After one turn, the task is finished, and the Turker's work is submitted
for review.

### Task 3: Multi-Agent Dialog

ParlAI supports dialogs between multiple agents, whether they are local
ParlAI agents or human Turkers. In the [Multi-Agent Dialog
task](https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk/tasks/multi_agent_dialog/),
one local human agents and two Turkers engage in a round-robin chat,
until the first local human agent sends a message ending with `[DONE]`,
after which other agents will send a final message and the task is
concluded.

This task uses the `MultiAgentDialogWorld` which is already implemented
in `parlai.core.worlds`.

### Task 4: Advanced Functionality - Deal or No Deal

ParlAI is able to support more than just generic chat. The [Deal or No
Deal
task](https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk/tasks/dealnodeal/)
provides additional functionality over the regular chat window to allow
users to view the items they are dividing, select an allocation, and
then submit a deal.

This task leverages the ability to override base functionality of the
core.html page using `task_config.py`. Javascript is added here to
replace the task description with additional buttons and UI elements
that are required for the more complicated task. These trigger within an
overridden handle\_new\_message function, which will only fire after an
agent has entered the chat. In general it is easier/preferred to use a
custom webpage as described in step 4 of "Creating Your Own Task",
though this is an alternate that can be used if you specifically only
want to show additional components in the task description pane of the
chat window.

### Task 5: Advanced Functionality - MTurk Qualification Flow

ParlAI MTurk is able to support filtering users through a form of
qualification system. The [Qualification Flow
task](https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk/tasks/qualification_flow_example)
demos this functionality using a simple "addition" task.

In this task, all users see a test version of the task on the first time
they enter it and a real version every subsequent time, however users
that fail to pass the test version are assigned a qualification that
prevents them from working on the task again. Thus ParlAI users are able
to filter out workers from the very beginning who don't necessarily meet
the specifications you are going for. This is preferred to filtering out
workers using the onboarding world for tasks that require a full
instance's worth of work to verify a worker's readiness.

### Task 6: Advanced Functionality - React Task Demo

ParlAI MTurk allows creation of arbitrary tasks, so long as the required
components can be created in React. The [React Task
Demo](https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk/tasks/react_task_demo)
task exists to show how this is set up for both cases where you are
building your own components from scratch and cases where you want to
import other components as dependancies.

This task consists of 3 agents participating in different roles with
different frontend needs. By setting `MTurkAgent.id` to the correct
values, different interfaces are displayed to an 'Asker' who can ask any
questions, an 'Answerer' who is only able to respond with numeric
values, and an Evaluator who observes the chat and approves or rejects
at the end. These components are defined and linked in the
`frontend/components/custom.jsx` file.

Creating Your Own Task
----------------------

ParlAI provides a generic MTurk dialog interface that you can use to
implement any kind of dialog task. To create your own task, start by
reading the README of the existing task that your task most resembles, and then subclass the appropriate components in order to write your own task. You may need to subclass the following:

### {{{TODO: revise this section onward}}}

{TODO: classes: ChatWorld, OnboardingWorld, BlueprintArgs, Blueprint}

If you are creating a new `Blueprint`, you will currently need to create a new `run.py` file in which to call your `Blueprint` so that its arguments can be read in correctly by Hydra; this should no longer be necessary as of the upcoming Hydra 1.1. You will likely need to specify the following helper files for your task:

{TODO: Helper files: conf/example.yaml, task_config/*}

A few things to keep in mind:

1.  To end a conversation, you should check to see if an action has
    `episode_done` set to `True`, as this signals that the world should
    start returning `True` for the `episode_done` function.
2.  Make sure to test your dialog task using MTurk's sandbox mode before
    pushing it live, by using the `--sandbox` flag (enabled by default)
    when running `run.py`.
3.  Your `worlds.py` worlds should be handling different types of agent
    disconnect messages. `MTurkAgent.act()` can return any of
    `MTURK_DISCONNECT_MESSAGE`, `RETURN_MESSAGE`, and `TIMEOUT_MESSAGE`
    as defined in `MTurkAgent`. Your world should still be able to
    continue to completion in any of these circumstances.
4.  NO DATA is saved automatically in the way that regular MTurk tasks
    save data. Unless you're using the Alpha saving and loading
    functionality described below, you'll need to save your data in your
    `world.shutdown()` function.

Advanced Task Techniques
------------------------

The Mephisto platform allows for a number of advanced customization
techniques to cover specialized tasks. See the [`bootstrap-chat` README](https://github.com/facebookresearch/Mephisto/blob/master/packages/bootstrap-chat/README.md) for a discussion of how to utilize Bootstrap-based UI components for crowdsourcing tasks.

Running a Task
--------------

If you have not used Mechanical Turk before, you will need an MTurk
Requester Account and an AWS account (these are two separate accounts).
Follow the steps below:

-   Sign up for an AWS account at
    [aws.amazon.com](https://aws.amazon.com/)
-   Sign up for an MTurk account at
    [requester.mturk.com](https://requester.mturk.com/)
-   Go to the developer tab
    ([<https://requester.mturk.com/developer>](https://requester.mturk.com/developer))
    and link your AWS account to your MTurk account (Step 2 on that
    screen)
-   MTurk also has a “Sandbox” which is a test version of the MTurk
    marketplace. You can use it to test publishing and completing tasks
    without paying any money. ParlAI supports the Sandbox. To use the
    Sandbox, you need to sign up for a [Sandbox
    account](http://requestersandbox.mturk.com/). You will then also
    need to [link your AWS
    account](http://requestersandbox.mturk.com/developer) to your
    Sandbox account. In order to test faster, you will also want to
    create a [Sandbox Worker account](http://workersandbox.mturk.com/).
    You can then view tasks that you publish from ParlAI and complete them
    yourself.
-   ParlAI's MTurk default functionality requires a free heroku account
    which can be obtained [here](https://signup.heroku.com/). Running
    any ParlAI MTurk operation will walk you through linking the two.
    
To run a crowdsourcing task, launch its run file (typically `run.py`) with the proper flags, using a command like the following:

```bash
python run.py \
mephisto.blueprint.num_conversations <num_conversations> \
mephisto.task.task_reward <reward> \
[mephisto.provider.requester_name=${REQUESTER_NAME} mephisto/architect=heroku]
```
For instance, to create 2 conversations for the [QA Data
Collection](https://github.com/facebookresearch/ParlAI/tree/master/parlai/crowdsourcing/tasks/qa_data_collection)
task with a reward of $0.05 per assignment in sandbox mode, run:

```bash
python parlai/crowdsourcing/tasks/qa_data_collection/run.py \
mephisto.blueprint.num_conversations 2 \
mephisto.task.task_reward 0.05
```

Make sure to test your task in MTurk sandbox mode first before pushing it live (with the `mephisto.provider.requester_name=${REQUESTER_NAME} mephisto/architect=heroku` flags).

Additional flags can be used for more specific purposes:

### {{{TODO: revise this section onward. Convert all existing flags, remove all unported ones, and maybe look if there are any other important ones to cover}}}

-   `--unique` ensures that an Turker is only able to complete one
    assignment, thus ensuring each assignment is completed by a unique
    person.
-   `--unique-qual-name <name>` lets you use the above functionality
    across more than one task. Workers will only be able to complete a
    task launched with this flag for a given &lt;name&gt; once.
-   `--allowed-conversations <num>` prevents a Turker from entering more
    than &lt;num&gt; conversations at once (by using multiple
    windows/tabs). This defaults to 0, which is unlimited.
-   `--count-complete` only counts completed assignments towards the
    num\_conversations requested. This may lead to more conversations
    being had than requested (and thus higher costs for instances where
    one Turker disconnects and we pay the other) but it ensures that if
    you request 1,000 conversations you end up with at least 1,000
    completed data points.
-   `--max-connections` controls the number of HITs can be launched at
    the same time. If not specified, it defaults to 30; 0 is unlimited.
-   `--max-time` sets a maximum limit in seconds for how many seconds
    per day a specific worker can work on your task. Data is logged to
    `working_time.pickle`, so all runs on the same machine will share
    the daily work logs.
-   `--max-time-qual` sets the specific qualification name for the
    max-time soft block. Using this can allow you to limit worker time
    between separate machines where `working_time.pickle` isn't shared

Reviewing Turker's Work
-----------------------

You can programmatically review work using the commands available in the
`CrowdTaskWorld` class: for example, see the sample code in the docstring of the [`.review_work()` method](https://github.com/facebookresearch/ParlAI/blob/master/parlai/crowdsourcing/utils/worlds.py) of that class. For instance, you can set HITs to be
automatically approved if they are deemed completed by the world.

If you don't take any action in 1 week, all HITs will be auto-approved
and Turkers will be paid.

ParlAI-MTurk Tips and Tricks
----------------------------

### Approving Work

-   Unless you explicitly set the `auto_approve_delay` argument in [`create_hit_type()`](https://github.com/facebookresearch/Mephisto/blob/master/mephisto/abstractions/providers/mturk/mturk_utils.py), or approve work by calling [`MTurkAgent.approve_work()`](https://github.com/facebookresearch/Mephisto/blob/master/mephisto/abstractions/providers/mturk/mturk_agent.py), work will be auto-approved after 7 days. Workers like getting paid quickly, so be mindful to not have too much delay before their HITs are approved.
-   Occasionally Turkers will take advantage of getting paid immediately
    without review if you auto approve their work by calling
    `MTurkAgent.approve_work()` at the close of the task. If you aren't
    using any kind of validation before you approve work or if you
    don't intend to review the work manually, consider relying on auto approval after a fixed time delay with the `auto_approve_delay` argument of `create_hit_type()` rather than approving immediately.

### Rejecting Work

-   Most Turkers take their work very seriously, so if you find yourself
    with many different workers making similar mistakes on your task,
    it's possible that the task itself is unclear. You __shouldn't__ be
    rejecting work in this case, but rather you should update your
    instructions and see if the problem is resolved.
-   Reject sparingly at first and give clear reasons for rejection / how
    to improve. Rejections with no context are a violation of Amazon's
    terms of service.

### Filtering Workers

-   For tasks for which it is reasonably easy to tell whether or not a
    worker is capable of working on the task (generally less than 5
    minutes of reading and interacting), it's appropriate to build a
    testing stage into your onboarding world. This stage should only be
    shown to workers once, and failing the task should soft-block the
    worker and expire the HIT.

### Soft-blocking vs. Hard-blocking

-   Soft-blocking workers who are clearly trying on a task but not
    __quite__ getting it allows those workers to work on other tasks for
    you in the future. You can soft block workers by calling [`Worker.grant_qualification()`](https://github.com/facebookresearch/Mephisto/blob/master/mephisto/data_model/qualification.py) for a certain `qualification_name`, which is typically set by the `mephisto.blueprint.block_qualification` parameter. That worker will not be able to work on any
    tasks that use the same `block_qualification`.
    
### Preventing and Handling Crashes

-   The `max_num_concurrent_units` argument when initializing [`TaskLauncher`](https://github.com/facebookresearch/Mephisto/blob/master/mephisto/operations/task_launcher.py) controls how many people can work on your task at any given time: set this sufficiently low for your task. Leaving this too high might leave your heroku server running
    into issues depending on how many messages per second it's trying to
    process, and on how much data is being sent in those messages (such
    as picture or video data).
-   If you're using a model on your local machine, try to share the
    model parameters whenever possible. Needing new parameters for each
    of your conversations might make your machine run out of memory, causing
    the data collection to crash in an manner that ParlAI can't handle.
-   If your task crashes, it's good to run [`mephisto/scripts/mturk/cleanup.py`](https://github.com/facebookresearch/Mephisto/blob/master/mephisto/scripts/mturk/cleanup.py) to find the task that had crashed and remove the orphan tasks.
-   If workers email you about task crashes with sufficient evidence
    that they were working on that task, offer to compensate them by sending
    them a bonus for their failed task on one of their other completed
    tasks, and bonus that HIT ID by calling [`MTurkWorker.bonus_worker()`](https://github.com/facebookresearch/Mephisto/blob/master/mephisto/abstractions/providers/mturk/mturk_worker.py).

### Task Design

-   Design and test your task using the developer sandbox feature (used
    by default when calling a run.py), only launch live mode (as detailed in the [crowdsourcing README](https://github.com/facebookresearch/ParlAI/tree/master/parlai/crowdsourcing#running-tasks-live)) after you've
    tested your flow entirely.
-   Launch a few small pilot HITs live before your main data
    collection, and manually review every response to see how well the
    workers are understanding your task. Use these test HITs to tweak your
    task instructions until you're satisfied with the results, as this
    will improve the quality of the received data.

### Other Tips

-   Check your MTurk-associated email account frequently when running a task,
    and be responsive to the workers working on your tasks. This is
    important to keep a good reputation in the MTurk community.
-   If you notice that certain workers are doing a really good job on
    the task, send them bonuses, as this will encourage them to work on
    your HITs more in the future. It will also be a visible way for you
    to acknowledge their good work.


Additional Credits
------------------

-   Turker icon credit: [Amazon Mechanical
    Turk](https://requester.mturk.com/).
-   Robot icon credit: [Icons8](https://icons8.com/).

