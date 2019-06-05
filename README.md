<p align="center"><img width="70%" src="docs/source/\_static/img/parlai.png" /></p>

[![CircleCI](https://circleci.com/gh/facebookresearch/ParlAI.svg?style=shield)](https://circleci.com/gh/facebookresearch/ParlAI/tree/master)

--------------------------------------------------------------------------------

[ParlAI](http://parl.ai) (pronounced “par-lay”) is a framework for dialogue AI research, implemented in Python.

Its goal is to provide researchers:

- a **unified framework** for sharing, training and testing dialogue models from open-domain chitchat to VQA.
- many popular **datasets available all in one place, with the same API**, among them [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [bAbI tasks](https://arxiv.org/abs/1502.05698), [MS MARCO](http://www.msmarco.org/), [WikiQA](https://www.microsoft.com/en-us/download/details.aspx?id=52419), SimpleQuestions](https://arxiv.org/abs/1506.02075), [WikiMovies](https://arxiv.org/abs/1606.03126), [QACNN & QADailyMail](https://arxiv.org/abs/1506.03340), [CBT](https://arxiv.org/abs/1511.02301), [BookTest](https://arxiv.org/abs/1610.00956), [bAbI Dialogue tasks](https://arxiv.org/abs/1605.07683), [Ubuntu Dialogue](https://arxiv.org/abs/1506.08909), [OpenSubtitles](http://opus.lingfil.uu.se/OpenSubtitles.php), [Cornell Movie](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), [VQA-COCO2014](http://visualqa.org/), [VisDial](https://arxiv.org/abs/1611.08669) and [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/) and more than 70 others. See the complete list [here](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/task_list.py)
- a wide set of **reference models** -- from retrieval baselines to Transformers.
- seamless **integration of [Amazon Mechanical Turk](https://www.mturk.com/mturk/welcome)** for data collection and human evaluation
- **integration with [Facebook Messenger](http://www.parl.ai/docs/tutorial_messenger.html)** to connect agents with humans in a chat interface

ParlAI is described in the following paper:
[“ParlAI: A Dialog Research Software Platform", arXiv:1705.06476](https://arxiv.org/abs/1705.06476). See the [news page](https://github.com/facebookresearch/ParlAI/blob/master/NEWS.md) for the latest additions & updates.

## Installing ParlAI

ParlAI currently requires Python3. Dependencies of the core modules are listed in requirement.txt. Some models included (in parlai/agents) have additional requirements.

Run the following commands to clone the repository and install ParlAI:

```bash
git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
cd ~/ParlAI; python setup.py develop
```

This will link the cloned directory to your site-packages.

This is the recommended installation procedure, as it provides ready access to the examples and allows you to modify anything you might need. This is especially useful if you if you want to submit another task to the repository.

All needed data will be downloaded to ~/ParlAI/data, and any non-data files (such as the MemNN code) if requested will be downloaded to ~/ParlAI/downloads. If you need to clear out the space used by these files, you can safely delete these directories and any files needed will be downloaded again.

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

- train and evaluate on live dialogue with humans via Mechanical Turk or Messenger
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
- Supports batch and hogwild training and evaluation of models.

## Documentation

 - [Basics: world, agents, teachers, action and observations](https://parl.ai/docs/tutorial_basic.html)
 - [Adding a dataset/task](http://www.parl.ai/docs/tutorial_task.html).
 - [Plug into MTurk](http://parl.ai/docs/tutorial_mturk.html)
 - [Plug into Facebook Messenger](http://parl.ai/docs/tutorial_messenger.html)

## Examples

A large set of examples can be found in [this directory](https://github.com/facebookresearch/ParlAI/tree/master/examples). Here are a few of them.
Note: If any of these examples fail, check the [requirements section](#requirements) to see if you have missed something.

Display 10 random examples from task 1 of the "1k training examples" bAbI task:
```bash
python examples/display_data.py -t babi:task1k:1
```

Evaluate an IR baseline model on the validation set of the Movies Subreddit dataset:
```bash
python examples/eval_model.py -m ir_baseline -t "#moviedd-reddit" -dt valid
```

Train a seq2seq model on the "10k training examples" bAbI task 1 with batch size of 32 examples until accuracy reaches 95% on validation (requires pytorch):
```bash
python examples/train_model.py -t babi:task10k:1 -m seq2seq -mf /tmp/model_s2s -bs 32 -vtim 30 -vcut 0.95
```


## Code

The code is set up into several main directories:

- **core**: contains the primary code for the framework
- **agents**: contains agents which can interact with the different tasks (e.g. machine learning models)
- **examples**: contains a few basic examples of different loops (building dictionary, train/eval, displaying data)
- **tasks**: contains code for the different tasks available from within ParlAI
- **mturk**: contains code for setting up Mechanical Turk, as well as sample MTurk tasks
- **messenger**: contains code for interfacing with Facebook Messenger
- **zoo**: contains code to directly download and use pretrained models from our model zoo

Each directory is described in more detail below, ordered by dependencies.

### Core

The core library contains the following files:

- **agents.py**: this file contains a few basic agents which can be extended by your own model
  - **_Agent_**: base class for all other agents, implements the act() method which receives an observation table and returns a table in response
  - **_Teacher_**: child of Agent, also implements the report method for returning metrics. Tasks implement the Teacher class
  - **_MultiTaskTeacher_**: creates a set of teachers based on a "task string" passed to the Teacher, creating multiple teachers within it and alternating between them
  - create_task_teacher: instantiate a teacher from a given task string (e.g. 'babi:task:1' or 'squad')
- **build_data.py**: basic utilities for setting up data for tasks. you can override if your filesystem needs different functionality.
- **dict.py**: contains code for building general NLP-style dictionaries from observations
  - DictionaryAgent: agent which tracks the index and frequency of words in a dictionary, and can parse a sentence into indices into its dictionary or back
- **metrics.py**: computes evaluation metrics, e.g. ranking metrics, etc.
- **params.py**: uses argparse to interpret command line arguments for ParlAI
- **teachers.py**: contains teachers that deal with dialogue-based tasks, as well as data classes for storing data
  - **_FixedDialogTeacher_**: base class for a teacher that utilizes fixed data
  - **_DialogTeacher_**: base class for a teacher doing dialogue with fixed chat logs
  - **_ParlAIDialogTeacher_**: a teacher that implements a simple standard text format for many tasks (non-visual tasks only)
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
Some agents currently available within [this directory](https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents):

- **drqa**: an attentive [LSTM model DrQA](https://arxiv.org/abs/1704.00051) implemented in PyTorch that has competitive results on the SQuAD dataset amongst others.
- **fairseq**: [an attentive sequence to sequence model using convolutions](https://arxiv.org/abs/1705.03122)
- **seq2seq**: a generic seq2seq model with various options
- **ibm_seq2seq**: IBM sequence to sequence model
- **language_model**: an RNN language model
- **memnn**: code for an end-to-end memory network
- **mlb_vqa**: a visual question answering model based on [this paper](https://arxiv.org/abs/1610.04325)
- **starspace**: a simple supervised embedding approach which is a strong baseline based on [this paper](https://arxiv.org/abs/1709.03856).
- **tfidf_retriever** a simple retrieval based model, also useful as a first step for retrieving information as input to another model.
- **ir_baseline**: simple information retrieval baseline that scores candidate responses with [TFIDF-weighted](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) matching
- **repeat_label**: basic class for merely repeating all data sent to it (e.g. for piping to a file, debugging)
- **remote_agent**: basic class for any agent connecting over ZMQ
- **local_human**: takes input from the keyboard as the act() function of the agent, so a human can act in the environment

See the [directory](https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents) for the complete list.



### Examples

[This directory](https://github.com/facebookresearch/ParlAI/tree/master/examples) contains a few particular examples of basic loops.

- base_train.py: _very simple example shows the outline of a training/validation loop using the default Agent parent class_
- display_data.py: _uses agent.repeat_label to display data from a particular task provided on the command-line_
- display_model.py: _shows the predictions of a provided model on a particular task provided on the command-line_
- eval_model.py: _uses the named agent to compute evaluation metrics data for a particular task provided on the command-line_
- build_dict.py: _build a dictionary from a particular task provided on the command-line using core.dict.DictionaryAgent_


## Support
If you have any questions, bug reports or feature requests, please don't hesitate to post on our [Github Issues page](https://github.com/facebookresearch/ParlAI/issues).

## The Team
ParlAI is currently maintained by Emily Dinan, Alexander H. Miller, Stephen Roller, Kurt Shuster, Jack Urbanek and Jason Weston.
A non-exhaustive list of other major contributors includes:
Will Feng, Adam Fisch, Jiasen Lu, Antoine Bordes, Devi Parikh, Dhruv Batra,
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
ParlAI is MIT licensed. See the LICENSE file for details.
