# ParlAI

ParlAI (pronounced “parley”) is a framework for dialog AI research, implemented in Python.

Its goal is to provide researchers a unified framework for training and testing of dialog models, including multi-task training over many datasets at once, as well as the seamless integration of <a href="https://www.mturk.com/mturk/welcome">Amazon Mechanical Turk</a> for data collection and human evaluation.

Over 20 tasks are supported in the first release, including popular datasets such as [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [bAbI tasks](https://arxiv.org/abs/1502.05698), [MCTest](https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/), [WikiQA](https://www.microsoft.com/en-us/download/details.aspx?id=52419), [WebQuestions](http://www.aclweb.org/anthology/D13-1160), [SimpleQuestions](https://arxiv.org/abs/1506.02075), [WikiMovies](https://arxiv.org/abs/1606.03126), [QACNN & QADailyMail](https://arxiv.org/abs/1506.03340), [CBT](https://arxiv.org/abs/1511.02301), [BookTest](https://arxiv.org/abs/1610.00956), [bAbI Dialog tasks](https://arxiv.org/abs/1605.07683), [Ubuntu Dialog](https://arxiv.org/abs/1506.08909), [OpenSubtitles](http://opus.lingfil.uu.se/OpenSubtitles.php), [Cornell Movie](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) and [VQA-COCO2014](http://visualqa.org/).

Included are examples of training neural models with [PyTorch](http://pytorch.org/) and [Lua Torch](http://torch.ch/), including both batch training on GPU and hogwild training on CPUs. Using [Theano](http://deeplearning.net/software/theano/) or [Tensorflow](https://www.tensorflow.org/) instead is also straightforward.

Our aim is for the number of tasks and agents that train on them to grow in a community-based way.

We are in an early-release Beta. Expect some adventures and rough edges.

## Basic Examples

`python examples/display_data.py -t babi:task1k:1`

Displays 10 random examples from task 1 of the "1000-example" bAbI task.

`python examples/memnn_luatorch_cpu/full_task_train.py -t cbt:NE -n 8`

Trains a simple cpu-based memory network on the Children's Book Test "Named Entities" subset with 8 threads (python processes) using Hogwild.

`python examples/drqa/train.py -t squad -b 32`

Trains an attentive LSTM model on the SQuAD dataset with a batch size of 32 examples.

## Requirements

ParlAI currently requires Python3.

Several models included (in parlai/agents) have additional requirements such as [PyTorch](http://pytorch.org/) or [Lua Torch](http://torch.ch/).

## Installing ParlAI

First, clone the repository, then enter the cloned directory.

Linked install:
Run `python setup.py develop` to link the cloned directory to your site-packages.
This is the recommended installation procedure if you plan on modifying any parlai code for your run or submitting a pull request, especially if you want to add another task to repository.
All needed data will be downloaded to ./data, and any model files (currently just the memnn model) if requested will be downloaded to ./downloads.

Copied install (use parlai only as a dependency):
Run `python setup.py install` to copy contents to your site-packages folder.
All data will be downloaded to that folder, and to make any changes to the code you will need to run install again.
If you want to just use parlai as a dependency (e.g. to access the tasks or the core code), this works fine.
If you want to clear out the downloaded data, then delete the 'data' and 'downloads' (if applicable) folder in site-packages/parlai.

## Goals

Unified framework for evaluation of dialogue models
- downloads tasks/datasets when requested and provides the same simple interface to them
- unify dataset input and evaluation frameworks/metrics
- agents/ directory encourages researchers to commit their training code to the repository to share with others
- aid reproducibility

End goal is general dialogue, which includes many different skills
- seamless combination of simulated and real language datasets
- encourage multi-task model development & evaluation
- reduce overfitting of models to specific datasets         

End goal is real dialogue with people
- train and evaluate on live dialogue with humans via MTurk
- easy setup for connecting turkers with your dialogue agent
- allow to compare different research groups turk experiments

Set of datasets to bootstrap a working dialogue model for human interaction
- motivates building new datasets that will go in the repository

## Properties

- All datasets look like natural dialogue: a single format / API.
- Both fixed datasets (conversation logs) and interactive (online/RL) tasks.
- Both real and simulated tasks.
- Supports other media, e.g. visual in VQA.
- Can use Mechanical Turk to run / collect data / evaluate.
- Python framework
- Examples of training with PyTorch.
- Uses zmq to talk to other toolboxes not in Python, examples of Lua Torch given.
- Supports hogwild and batch training of models.

## Worlds, agents and teachers
The main concepts (classes) in ParlAI:
- world - defines the environment (can be very simple, just two agents talking to each other).
- agent – an agent in the world, e.g. the learner. (There can be multiple learners.)
- teacher – a type of agent that talks to the learner, implements one of the tasks listed before.

After defining a world, and the agents in it, a main loop can be run for training, testing or displaying which calls the function world.parley(). The skeleton of an example main is given in the left panel, and the actual code for parley() on the right.

<p align=center><img width="100%" src="docs/source/\_static/img/main.png" /></p>


## Actions and Observations

All agents (including teachers) speak to each other with a single format -- the observation/action object (a python dict).
This is used to pass text, labels and rewards between agents.
It’s the same object type when talking (acting) or listening (observing), but a different view (with different values in the fields).
The fields are as follows:

<p align=center><img width="33%" src="docs/source/\_static/img/act-obs-dict.png" /></p>


Each of these fields are technically optional, depending on your dataset, though the 'text' field will most likely be used in nearly all exchanges.

For a fixed supervised learning dataset like bAbI, a typical exchange from the training set might be as follows (the test set would not include labels):

```python
Teacher: {
    'text': 'Sam went to the kitchen\nPat gave Sam the milk\nWhere is the milk?',
    'labels': ['kitchen'],
    'label_candidates': ['hallway', 'kitchen', 'bathroom'],
    'episode_done': False
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
```

## Code

The code is set up into several main directories:

- **core**: contains the primary code for the framework
- **agents**: contains agents which can interact with the different tasks (e.g. machine learning models)
- **examples**: contains a few basic examples of different loops (building dictionary, train/eval, displaying data)
- **tasks**: contains code for the different tasks available from within ParlAI

Each directory is described in more detail below, ordered by dependencies.

### Core

The core library contains the following files:

- **agents.py**: this file contains a few basic agents which can be extended by your own model
  - **_Agent_**: base class for all other agents, implements the act() method which receives an observation table and returns a table in response
  - **_Teacher_**: child of Agent, also implements the report method for returning metrics. Tasks implement the Teacher class
  - **_MultiTaskTeacher_**: creates a set of teachers based on a "task string" passed to the Teacher, creating multiple teachers within it and alternating between them
  - create_task_teacher: instantiate a teacher from a given task string (e.g. 'babi:task:1' or 'squad')
- **build_data.py**: basic utilities for setting up data for tasks. you can override if your filesystem needs different functionality.
- **data.py**: contains some default classes for fixed text datasets
  - TextData: sets up observation tables with 'text', 'labels', 'reward', and/or 'candidates' fields
  - HogwildTextData: does the same thing as TextData, but stores underlying data in a shared-memory array
- **dialog_teacher.py**: contains a base teacher class for doing dialog with fixed chat logs
- **dict.py**: contains code for building general NLP-style dictionaries from observations
  - DictionaryAgent: agent which tracks the index and frequency of words in a dictionary, and can parse a sentence into indices into its dictionary or back
- **fbdialog_teacher.py**: contains a teacher class which implements a function setup_data which parses data in the FB Dialog data format
- **metrics.py**: computes evaluation metrics for dialog, e.g. ranking metrics, etc.
- **params.py**: uses argparse to interpret command line arguments for ParlAI
- **thread_utils.py**: utility classes/functions for use in Hogwild multithreading (multiprocessing)
  - SharedTable: provides a lock-protected, shared-memory, dictionary-like interface for keeping track of metrics
- **worlds.py**: contains a set of basic worlds for tasks to take place inside
  - **_World_**: base class for all other worlds, implements `parley`, `shutdown`, `__enter__`, and `__exit__`
  - **_DialogPartnerWorld_**: default world for turn-based two-agent communication
        MultiAgentDialogWorld: round-robin turn-based agent communication for two or more agents
        HogwildWorld: default world for setting up a separate world for every thread when using multiple threads (processes)


### Agents

The agents directory contains agents that have been approved into the ParlAI framework for shared use.
Currently availabe within this directory:

- **drqa**: an attentive LSTM model DrQA (https://arxiv.org/abs/1704.00051) implemented in PyTorch that has competitive results on the SQuAD dataset amongst others.
- **memnn**: code for an end-to-end memory network in Lua Torch
- **remote_agent**: basic class for any agent connecting over ZMQ (memnn_luatorch_cpu uses this)
- **ir_baseline**: simple information retrieval baseline that scores candidate responses with TFIDF-weighted matching
- **repeat_label**: basic class for merely repeating all data sent to it (e.g. for piping to a file, debugging)

### Examples

This directory contains a few particular examples of basic loops.

- display_data.py: _uses agent.repeat_label to display data from a particular task provided on the command-line_
- display_model.py: _shows the predictions of a provided model on a particular task provided on the command-line_
- eval_model.py: _uses agent.repeat_label to compute evaluation metrics data for a particular task provided on the command-line_
- build_dict.py: _build a dictionary from a particular task provided on the command-line using core.dict.DictionaryAgent_
- memnn_luatorch_cpu: _shows a few examples of training an end-to-end memory network on a few datasets_
- drqa: _shows how to train the attentive LSTM DrQA model of <a href="https://arxiv.org/abs/1704.00051">Chen et al.</a> on SQuAD._

### Tasks


Over 20 tasks are supported in the first release, including popular datasets such as
SQuAD, bAbI tasks, MCTest, WikiQA, WebQuestions, SimpleQuestions, WikiMovies, QACNN, QADailyMail, CBT, BookTest, bAbI Dialog tasks,
Ubuntu, OpenSubtitles, Cornell Movie and VQA-COCO2014.

Our first release includes the following datasets (shown in the left panel), and accessing one of them is as simple as specifying the name of the task as a command line option, as shown in the dataset display utility (right panel):
<p align=center><img width="100%" src="docs/source/\_static/img/tasks.png" /></p>

See <a href="https://github.com/fairinternal/ParlAI/tree/master/parlai/tasks/tasks.json">here</a> for the current complete task list.

Choosing a task in ParlAI is as easy as specifying it on the command line, as shown in the above image (right). If the dataset has not been used before, ParlAI will automatically download it. As all datasets are treated in the same way in ParlAI (with a single dialog API), a dialog agent can in principle switch training and testing between any of them. Even better, one can specify many tasks at once (multi-tasking) by simply providing a comma-separated list, e.g.  the command line “-t babi,squad”, to use those two datasets, or even all  the QA datasets at once  (-t #qa) or indeed every task in ParlAI at once (-t #all). The aim is to make it easy to build and evaluate very rich dialog models.


Each task folder contains:
- **build.py** file for setting up data for the task (downloading data, etc, only done the first time requested, and not downloaded if the task is not used).
- **agents.py** file which contains default or special teacher classes used by core.create_task to instantiate these classes from command-line arguments (if desired).
- **worlds.py** file can optionally be added for tasks that need to define new/complex environments.

To add your own task:
- (optional) implement build.py to download any needed data
- implement agents.py, with at least a DefaultTeacher (extending Teacher or one of its children)
    - if your data is in FB Dialog format, subclass FbDialogTeacher
    - if not...
        - if your data is text-based, you can use extend DialogTeacher and thus core.data.TextData, in which case you just need to write your own setup_data function which provides an iterable over the data according to the format described in core.data
        - if your data uses other fields, write your own act() method which provides observations from your task each time it's called

## License
ParlAI is BSD-licensed. We also provide an additional patent grant.
