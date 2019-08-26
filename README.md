<p align="center"><img width="70%" src="docs/source/\_static/img/parlai.png" /></p>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/ParlAI/blob/master/LICENSE) [![CircleCI](https://circleci.com/gh/facebookresearch/ParlAI.svg?style=shield)](https://circleci.com/gh/facebookresearch/ParlAI/tree/master) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/ParlAI/blob/master/CONTRIBUTING.md) [![Twitter](https://img.shields.io/twitter/follow/parlai_parley?label=Twitter&style=social)](https://twitter.com/parlai_parley)

--------------------------------------------------------------------------------

[ParlAI](http://parl.ai) (pronounced “par-lay”) is a python framework for
sharing, training and testing dialogue models, from open-domain chitchat to
VQA (Visual Question Answering).

Its goal is to provide researchers:

- **70+ popular datasets available all in one place, with the same API**, among them [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [MS MARCO](http://www.msmarco.org/), [QuAC](https://www.aclweb.org/anthology/D18-1241), [HotpotQA](https://hotpotqa.github.io/), [QACNN & QADailyMail](https://arxiv.org/abs/1506.03340), [CBT](https://arxiv.org/abs/1511.02301), [BookTest](https://arxiv.org/abs/1610.00956), [bAbI Dialogue tasks](https://arxiv.org/abs/1605.07683), [Ubuntu Dialogue](https://arxiv.org/abs/1506.08909), [PersonaChat](https://arxiv.org/abs/1801.07243), [OpenSubtitles](http://opus.lingfil.uu.se/OpenSubtitles.php), [Wizard of Wikipedia](https://openreview.net/forum?id=r1l73iRqKm), [VQA-COCO2014](http://visualqa.org/), [VisDial](https://arxiv.org/abs/1611.08669) and [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/). See the complete list [here](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/task_list.py)
- a wide set of **reference models** -- from retrieval baselines to transformers.
- a large zoo of **pretrained models** ready to use off-the-shelf
- seamless **integration of [Amazon Mechanical Turk](https://www.mturk.com/mturk/welcome)** for data collection and human evaluation
- **integration with [Facebook Messenger](http://www.parl.ai/docs/tutorial_messenger.html)** to connect agents with humans in a chat interface
- a large range of **helpers to create your own agents** and train on several tasks with **multitasking**
- **multimodality**, some tasks use text and images

ParlAI is described in the following paper:
[“ParlAI: A Dialog Research Software Platform", arXiv:1705.06476](https://arxiv.org/abs/1705.06476).

See the [news page](https://github.com/facebookresearch/ParlAI/blob/master/NEWS.md) for the latest additions & updates, and the website [http://parl.ai](http://parl.ai) for further docs.

## Installing ParlAI

ParlAI currently requires Python3 and [Pytorch](https://pytorch.org) 1.1 or
newer. Dependencies of the core modules are listed in `requirement.txt`. Some
models included (in `parlai/agents`) have additional requirements.

Run the following commands to clone the repository and install ParlAI:

```bash
git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
cd ~/ParlAI; python setup.py develop
```

This will link the cloned directory to your site-packages.

This is the recommended installation procedure, as it provides ready access to the examples and allows you to modify anything you might need. This is especially useful if you if you want to submit another task to the repository.

All needed data will be downloaded to `~/ParlAI/data`, and any non-data files if requested will be downloaded to `~/ParlAI/downloads`. If you need to clear out the space used by these files, you can safely delete these directories and any files needed will be downloaded again.

## Documentation

 - [Quick Start](https://parl.ai/docs/tutorial_quick.html)
 - [Basics: world, agents, teachers, action and observations](https://parl.ai/docs/tutorial_basic.html)
 - [List of available tasks/datasets](https://parl.ai/docs/tasks.html)
 - [Creating a dataset/task](http://www.parl.ai/docs/tutorial_task.html)
 - [List of available agents](./parlai/agents)
 - [Creating a new agent](https://parl.ai/docs/tutorial_seq2seq.html#)
 - [Model zoo (pretrained models)](https://parl.ai/docs/zoo.html)
 - [Plug into MTurk](http://parl.ai/docs/tutorial_mturk.html)
 - [Plug into Facebook Messenger](http://parl.ai/docs/tutorial_messenger.html)


## Examples

A large set of examples can be found in [this directory](./examples). Here are a few of them.
Note: If any of these examples fail, check the [requirements section](#requirements) to see if you have missed something.

Display 10 random examples from the SQuAD task
```bash
python examples/display_data.py -t squad
```

Evaluate an IR baseline model on the validation set of the Personachat task:
```bash
python examples/eval_model.py -m ir_baseline -t personachat -dt valid
```

Train a single layer transformer on personachat (requires pytorch and torchtext).
Detail: embedding size 300, 4 attention heads,  2 epochs using batchsize 64, word vectors are initialized with fasttext and the other elements of the batch are used as negative during training.
```bash
python examples/train_model.py -t personachat -m transformer/ranker -mf /tmp/model_tr6 --n-layers 1 --embedding-size 300 --ffn-size 600 --n-heads 4 --num-epochs 2 -veps 0.25 -bs 64 -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc --candidates batch
```



## Code Organization

The code is set up into several main directories:

- [**core**](./parlai/core): contains the primary code for the framework
- [**agents**](./parlai/agents): contains agents which can interact with the different tasks (e.g. machine learning models)
- [**examples**](./parlai/examples): contains a few basic examples of different loops (building dictionary, train/eval, displaying data)
- [**tasks**](./parlai/tasks): contains code for the different tasks available from within ParlAI
- [**mturk**](./parlai/mturk): contains code for setting up Mechanical Turk, as well as sample MTurk tasks
- [**messenger**](./parlai/messenger): contains code for interfacing with Facebook Messenger
- [**zoo**](./parlai/zoo): contains code to directly download and use pretrained models from our model zoo

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
