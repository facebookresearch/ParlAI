<p align="center"><img width="70%" src="docs/source/\_static/img/parlai.png" /></p>

--------------------------------------------------------------------------------

ParlAI (pronounced “par-lay”) is a framework for dialog AI research, implemented in Python.

Its goal is to provide researchers:
- a unified framework for sharing, training and testing dialog models
- multi-task training over many datasets at once
- seamless integration of [Amazon Mechanical Turk](https://www.mturk.com/mturk/welcome) for data collection and human evaluation

Over 20 [tasks](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/task_list.py) are currently supported, including popular datasets such as [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [bAbI tasks](https://arxiv.org/abs/1502.05698), [MS MARCO](http://www.msmarco.org/), [MCTest](https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/), [WikiQA](https://www.microsoft.com/en-us/download/details.aspx?id=52419), [WebQuestions](http://www.aclweb.org/anthology/D13-1160), [SimpleQuestions](https://arxiv.org/abs/1506.02075), [WikiMovies](https://arxiv.org/abs/1606.03126), [QACNN & QADailyMail](https://arxiv.org/abs/1506.03340), [CBT](https://arxiv.org/abs/1511.02301), [BookTest](https://arxiv.org/abs/1610.00956), [bAbI Dialog tasks](https://arxiv.org/abs/1605.07683), [Ubuntu Dialog](https://arxiv.org/abs/1506.08909), [OpenSubtitles](http://opus.lingfil.uu.se/OpenSubtitles.php), [Cornell Movie](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), [VQA-COCO2014](http://visualqa.org/), [VisDial](https://arxiv.org/abs/1611.08669) and [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/). See [here](http://www.parl.ai/static/docs/tasks.html#) for the current complete task list.

Included are examples of training neural models with [PyTorch](http://pytorch.org/) and [Lua Torch](http://torch.ch/), with batch training on GPU or hogwild training on CPUs. Using [Theano](http://deeplearning.net/software/theano/) or [Tensorflow](https://www.tensorflow.org/) instead is also straightforward.

Our aim is for the number of tasks and agents that train on them to grow in a community-based way.

ParlAI is described in the following paper:
[“ParlAI: A Dialog Research Software Platform", arXiv:1705.06476](https://arxiv.org/abs/1705.06476).


We are in an early-release Beta. Expect some adventures and rough edges.<br>
See the [news page](https://github.com/facebookresearch/ParlAI/blob/master/NEWS.md) for the latest additions & updates, and the website [http://parl.ai](http://parl.ai) for further docs.

Please also note there is a [ParlAI Request For Proposals funding university teams, 7 awards are available - deadline Aug 25.](https://research.fb.com/programs/research-awards/proposals/parlai/)

## Goals

Unified framework for evaluation of dialogue models
- downloads tasks/datasets on demand and provides the same simple interface to them
- unifies dataset input and evaluation frameworks/metrics
- `agents/` directory encourages researchers to submit their training code to the repository to share with others
- aids reproducibility

End goal is general dialogue, which includes many different skills
- seamlessly combines simulated and real language tasks
- encourages multi-task model development & evaluation
- helps to reduce overfitting of models to specific datasets         

End goal is real dialogue with people
- train and evaluate on live dialogue with humans via Mechanical Turk
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
- Python framework.
- Examples of training with PyTorch.
- Uses zmq to talk to other toolboxes not in Python, examples of Lua Torch given.
- Supports hogwild and batch training of models.

## Basic Examples

Note: If any of these examples fail, check the [requirements section](#requirements) to see if you have missed something.

Display 10 random examples from task 1 of the "1k training examples" bAbI task:
```bash
python examples/display_data.py -t babi:task1k:1
```

Displays 100 random examples from multi-tasking on the bAbI task and the SQuAD dataset at the same time:
```bash
python examples/display_data.py -t babi:task1k:1,squad -n 100
```

Evaluate on the bAbI test set with a human agent (using the local keyboard as input):
```bash
python examples/eval_model.py -m local_human -t babi:Task1k:1 -dt valid
```

Evaluate an IR baseline model on the validation set of the Movies Subreddit dataset:
```bash
python examples/eval_model.py -m ir_baseline -t "#moviedd-reddit" -dt valid
```

Display the predictions of that same IR baseline model:
```bash
python examples/display_model.py -m ir_baseline -t "#moviedd-reddit" -dt valid
```

Train a seq2seq model on the "1k training examples" bAbI task 1 with batch size of 8 examples for one epoch (requires pytorch):
```bash
python examples/train_model.py -m seq2seq -t babi:task1k:1 -bs 8 -e 1 -mf /tmp/model_s2s
```

Trains an attentive LSTM model on the SQuAD dataset with a batch size of 32 examples (pytorch and regex):
```bash
python examples/train_model.py -m drqa -t squad -bs 32 -mf /tmp/model_drqa
```

## Requirements

ParlAI currently requires Python3.

Dependencies of the core modules are listed in requirement.txt.

Several models included (in parlai/agents) have additional requirements.
DrQA requires installing [PyTorch](http://pytorch.org/), and the MemNN model requires installing [Lua Torch](http://torch.ch/docs/getting-started.html). See their respective websites for installation instructions.

## Installing ParlAI

Run the following commands to clone the repository and install ParlAI:

```bash
git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
cd ~/ParlAI; python setup.py develop
```

This will link the cloned directory to your site-packages.

This is the recommended installation procedure, as it provides ready access to the examples and allows you to modify anything you might need. This is especially useful if you if you want to submit another task to the repository.

All needed data will be downloaded to ~/ParlAI/data, and any non-data files (such as the MemNN code) if requested will be downloaded to ~/ParlAI/downloads. If you need to clear out the space used by these files, you can safely delete these directories and any files needed will be downloaded again.

## Worlds, agents and teachers

The main concepts (classes) in ParlAI:
- world - defines the environment (can be very simple, just two agents talking to each other).
- agent – an agent in the world, e.g. the learner. (There can be multiple learners.)
- teacher – a type of agent that talks to the learner, implements one of the 
listed before.

After defining a world and the agents in it, a main loop can be run for training, testing or displaying, which calls the function world.parley(). The skeleton of an example main is given in the left panel, and the actual code for parley() on the right.

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
- **mturk**: contains code for setting up Mechanical Turk, as well as sample MTurk tasks

Each directory is described in more detail below, ordered by dependencies.

### Core

The core library contains the following files:

- **agents.py**: this file contains a few basic agents which can be extended by your own model
  - **_Agent_**: base class for all other agents, implements the act() method which receives an observation table and returns a table in response
  - **_Teacher_**: child of Agent, also implements the report method for returning metrics. Tasks implement the Teacher class
  - **_MultiTaskTeacher_**: creates a set of teachers based on a "task string" passed to the Teacher, creating multiple teachers within it and alternating between them
  - create_task_teacher: instantiate a teacher from a given task string (e.g. 'babi:task:1' or 'squad')
- **build_data.py**: basic utilities for setting up data for tasks. you can override if your filesystem needs different functionality.
- **dialog_teacher.py**: contains a base teacher class for doing dialog with fixed chat logs, along with a data class for storing the data
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
  - **_MultiAgentDialogWorld_**: round-robin turn-based agent communication for two or more agents
  - **_HogwildWorld_**: default world for setting up a separate world for every thread when using multiple threads (processes)


### Agents

The agents directory contains agents that have been approved into the ParlAI framework for shared use.
We encourage you to contribute new ones!
Currently available within this directory:

- **drqa**: an attentive [LSTM model DrQA](https://arxiv.org/abs/1704.00051) implemented in PyTorch that has competitive results on the SQuAD dataset amongst others.
- **memnn**: code for an end-to-end memory network in Lua Torch
- **remote_agent**: basic class for any agent connecting over ZMQ (memnn_luatorch_cpu uses this)
- **ir_baseline**: simple information retrieval baseline that scores candidate responses with [TFIDF-weighted](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) matching
- **repeat_label**: basic class for merely repeating all data sent to it (e.g. for piping to a file, debugging)
- **local_human**: takes input from the keyboard as the act() function of the agent, so a human can act in the environment

### Examples

This directory contains a few particular examples of basic loops.

- base_train.py: _very simple example shows the outline of a training/validation loop using the default Agent parent class_
- display_data.py: _uses agent.repeat_label to display data from a particular task provided on the command-line_
- display_model.py: _shows the predictions of a provided model on a particular task provided on the command-line_
- eval_model.py: _uses the named agent to compute evaluation metrics data for a particular task provided on the command-line_
- build_dict.py: _build a dictionary from a particular task provided on the command-line using core.dict.DictionaryAgent_
- memnn_luatorch_cpu: _shows a few examples of training an end-to-end memory network on a few datasets_
- drqa: _shows how to train the attentive LSTM DrQA model of [Chen et al.](https://arxiv.org/abs/1704.00051) on SQuAD._

### Tasks

Our first release included the following datasets (shown in the left panel), and accessing one of them is as simple as specifying the name of the task as a command line option, as shown in the dataset display utility (right panel):
<p align=center><img width="100%" src="docs/source/\_static/img/tasks.png" /></p>

Over 20 tasks were supported in the first release, including popular datasets such as
SQuAD, bAbI tasks, MCTest, WikiQA, WebQuestions, SimpleQuestions, WikiMovies, QACNN, QADailyMail, CBT, BookTest, bAbI Dialog tasks,
Ubuntu, OpenSubtitles, Cornell Movie, VQA-COCO2014.
Since then, several datasets have been added such as  VQAv2, VisDial, MNIST_QA, Personalized Dialog, InsuranceQA, MS MARCO, TriviaQA, and CLEVR. See [here](http://www.parl.ai/static/docs/tasks.html#) for the current complete task list.

Choosing a task in ParlAI is as easy as specifying it on the command line, as shown in the above image (right). If the dataset has not been used before, ParlAI will automatically download it. As all datasets are treated in the same way in ParlAI (with a single dialog API), a dialog agent can in principle switch training and testing between any of them. Even better, one can specify many tasks at once (multi-tasking) by simply providing a comma-separated list, e.g.  the command line “-t babi,squad”, to use those two datasets, or even all  the QA datasets at once  (-t #qa) or indeed every task in ParlAI at once (-t #all). The aim is to make it easy to build and evaluate very rich dialog models.


Each task folder contains:
- **build.py** file for setting up data for the task (downloading data, etc, only done the first time requested, and not downloaded if the task is not used).
- **agents.py** file which contains default or special teacher classes used by core.create_task to instantiate these classes from command-line arguments (if desired).
- **worlds.py** file can optionally be added for tasks that need to define new/complex environments.

To add your own task:
- (optional) implement build.py to download any needed data
- implement agents.py, with at least a DefaultTeacher (extending Teacher or one of its children)
    - if your data is in [FB Dialog format](https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/fbdialog_teacher.py), subclass FbDialogTeacher
    - if not...
        - if your data consists of fixed logs, you can use extend DialogTeacher and thus core.data.TextData, in which case you just need to write your own setup_data function which provides an iterable over the data according to the format described in core.data
        - if your data uses other fields, write your own act() method which provides observations from your task each time it's called

### MTurk

An important part of ParlAI is seamless integration with Mechanical Turk for data collection, training and evaluation.

Human Turkers are also viewed as agents in ParlAI and hence person-person, person-bot, or multiple people and bots in group chat can all converse within the standard framework, switching out the roles as desired with no code changes to the agents. This is because Turkers also receive and send via a (pretty printed) version of the same interface, using the fields of the observation/action dict.

We currently provide three examples: collecting data, human evaluation of a bot, and round-robin chat between local humans and remote Turkers.

<p align=center><img width="100%" src="docs/source/\_static/img/mturk.png" /></p>

The mturk library contains the following directories:

- **core**: this directory contains the core code for setting up AWS backend that supports the MTurk chat interface, code for HIT creation and approval, and the wrapper class `MTurkAgent` which encapsulates the MTurk interface into a standard `Agent` class.
- **tasks**: this directory contains three sample MTurk tasks.
  - **_qa\_data\_collection_**: get questions and answers from turkers, given a random paragraph from SQuAD.
  - **_model\_evaluator_**: ask turkers to evaluate the information retrieval baseline model on the Reddit movie dialog dataset.
  - **_multi\_agent\_dialog_**: round-robin chat between two local human agents and two Turkers.

To run an MTurk task:
- Go into the directory for the task you want to run.
- Run `python run.py -nh <num_hits> -na <num_assignments> -r <reward> [--sandbox]/[--live]`, with `<num_hits>`, `<num_assignments>` and `<reward>` set appropriately. Use `--sandbox` to run the task in MTurk sandbox mode before pushing it live.

To add your own MTurk task:
- create a new folder within the mturk/tasks directory for your new task
- implement __task\_config.py__, with at least the following fields in the `task_config` dictionary:
  - `hit_title`: a short and descriptive title about the kind of task the HIT contains. On the Amazon Mechanical Turk web site, the HIT title appears in search results, and everywhere the HIT is mentioned.
  - `hit_description`: a description includes detailed information about the kind of task the HIT contains. On the Amazon Mechanical Turk web site, the HIT description appears in the expanded view of search results, and in the HIT and assignment screens.
  - `hit_keywords`: one or more words or phrases that describe the HIT, separated by commas. On MTurk website, these words are used in searches to find HITs.
  - `task_description`: a detailed task description that will be shown on the HIT task preview page and on the left side of the chat page. Supports HTML formatting.
- implement __run.py__, with code for setting up and running the world where `MTurkAgent` lives in.
- (Optional) implement __worlds.py__, with a world class that extends from `World`.

Please see [the MTurk tutorial](http://parl.ai/static/docs/mturk.html) to learn more about the MTurk examples and how to create and run your own task.

## Support
If you have any questions, bug reports or feature requests, please don't hesitate to post on our [Github Issues page](https://github.com/facebookresearch/ParlAI/issues).

## The Team
ParlAI is currently maintained by Alexander H. Miller, Jack Urbanek and Jason Weston.
A non-exhaustive list of other major contributors includes:
Will Feng, Adam Fisch,  Jiasen Lu, Antoine Bordes, Devi Parikh, Dhruv Batra,
Filipe de Avila Belbute Peres and Chao Pan.

## Citation

Please cite the [arXiv paper](https://arxiv.org/abs/1705.06476) if you use ParlAI in your work:

```
@article{miller2017parlai,
  title={ParlAI: A Dialog Research Software Platform},
  author={{Miller}, A.~H. and {Feng}, W. and {Fisch}, A. and {Lu}, J. and {Batra}, D. and {Bordes}, A. and {Parikh}, D. and {Weston}, J.},
  journal={arXiv preprint arXiv:{1705.06476}},
  year={2017}
}
```

## License
ParlAI is BSD-licensed. We also provide an additional patent grant.
