<p align="center">
 <img width="70%" src="docs/source/\_static/img/parlai.png" />
</p>

<p align="center">
   <a href="https://github.com/facebookresearch/ParlAI/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="CircleCI" />
  </a>
   <a href="https://pypi.org/project/parlai/">
    <img src="https://img.shields.io/pypi/v/parlai?color=blue&label=release" alt="CircleCI" />
  </a>
    <a href="https://circleci.com/gh/facebookresearch/ParlAI/tree/main">
    <img src="https://img.shields.io/circleci/build/github/facebookresearch/ParlAI/main" alt="Coverage" />
  </a>
    <a href="https://codecov.io/gh/facebookresearch/ParlAI">
    <img src="https://img.shields.io/codecov/c/github/facebookresearch/ParlAI" alt="GitHub contributors" />
  </a>
    <a href="https://img.shields.io/github/contributors/facebookresearch/ParlAI">
    <img src="https://img.shields.io/github/contributors/facebookresearch/ParlAI"/>
  </a>
    <a href="https://twitter.com/parlai_parley">
    <img src="https://img.shields.io/twitter/follow/parlai_parley?label=Twitter&style=social" alt="Twitter" />
  </a>
 </p>
 
-------------------------------------------------------------------------------------------------------------------------------------------------------

[ParlAI](http://parl.ai) (pronounced “par-lay”) is a python framework for
sharing, training and testing dialogue models, from open-domain chitchat, to
task-oriented dialogue, to visual question answering.

Its goal is to provide researchers:

- **100+ popular datasets available all in one place, with the same API**, among them [PersonaChat](https://arxiv.org/abs/1801.07243), [DailyDialog](https://arxiv.org/abs/1710.03957), [Wizard of Wikipedia](https://openreview.net/forum?id=r1l73iRqKm), [Empathetic Dialogues](https://arxiv.org/abs/1811.00207), [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [MS MARCO](http://www.msmarco.org/), [QuAC](https://www.aclweb.org/anthology/D18-1241), [HotpotQA](https://hotpotqa.github.io/), [QACNN & QADailyMail](https://arxiv.org/abs/1506.03340), [CBT](https://arxiv.org/abs/1511.02301), [BookTest](https://arxiv.org/abs/1610.00956), [bAbI Dialogue tasks](https://arxiv.org/abs/1605.07683), [Ubuntu Dialogue](https://arxiv.org/abs/1506.08909), [OpenSubtitles](http://opus.lingfil.uu.se/OpenSubtitles.php),  [Image Chat](https://arxiv.org/abs/1811.00945), [VQA](http://visualqa.org/), [VisDial](https://arxiv.org/abs/1611.08669) and [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/). See the complete list [here](https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/task_list.py).
- a wide set of [**reference models**](https://parl.ai/docs/agents_list.html) -- from retrieval baselines to Transformers.
- a large [zoo of **pretrained models**](https://parl.ai/docs/zoo.html) ready to use off-the-shelf
- seamless **integration of [Amazon Mechanical Turk](https://www.mturk.com/mturk/welcome)** for data collection and human evaluation
- **integration with [Facebook Messenger](https://parl.ai/docs/tutorial_chat_service.html)** to connect agents with humans in a chat interface
- a large range of **helpers to create your own agents** and train on several tasks with **multitasking**
- **multimodality**, some tasks use text and images


ParlAI is described in the following paper:
[“ParlAI: A Dialog Research Software Platform", arXiv:1705.06476](https://arxiv.org/abs/1705.06476)
or see these [more up-to-date slides](https://drive.google.com/file/d/1JfUW4AVrjSp8X8Fp0_rTTRoLxUfW0aUm/view?usp=sharing).

Follow us on [Twitter](https://twitter.com/parlai_parley) and check out our [Release
notes](https://github.com/facebookresearch/ParlAI/releases) to see the latest
information about new features & updates, and the website
[http://parl.ai](http://parl.ai) for further docs. For an archived list of updates,
check out [NEWS.md](https://github.com/facebookresearch/ParlAI/blob/main/NEWS.md).

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/facebookresearch/ParlAI/main/docs/source/_static/img/parlai_example.png" /></p>

## Interactive Tutorial

For those who want to start with ParlAI now, you can try our [Colab Tutorial](https://colab.research.google.com/drive/1bRMvN0lGXaTF5fuTidgvlAl-Lb41F7AD#scrollTo=KtVz5dCUmFkN).

## Installing ParlAI

ParlAI currently requires Python3.8+ and [Pytorch](https://pytorch.org) 1.6 or higher.
Dependencies of the core modules are listed in [`requirements.txt`](https://github.com/facebookresearch/ParlAI/blob/main/requirements.txt). Some
models included (in [`parlai/agents`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agents)) have additional requirements.
We *strongly* recommend you install ParlAI in a [venv](https://docs.python.org/3/library/venv.html) or [conda](https://www.anaconda.com/) environment.

We do not support Windows at this time, but many users [report success on Windows using Python 3.8](https://github.com/facebookresearch/ParlAI/issues/3989) and issues with Python 3.9. We are happy to accept patches that improve Windows support.

**Standard Installation**

If you want to use ParlAI without modifications, you can install it with:

```bash
pip install parlai
```

**Development Installation**

Many users will want to modify some parts of ParlAI. To set up a development
environment, run the following commands to clone the repository and install
ParlAI:

```bash
git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
cd ~/ParlAI; python setup.py develop
```

All needed data will be downloaded to `~/ParlAI/data`. If you need to clear out
the space used by these files, you can safely delete these directories and any
files needed will be downloaded again.

## Documentation

 - [Quick Start](https://parl.ai/docs/tutorial_quick.html)
 - [Basics: world, agents, teachers, action and observations](https://parl.ai/docs/tutorial_basic.html)
 - [Creating a new dataset/task](http://parl.ai/docs/tutorial_task.html)
 - [List of available tasks/datasets](https://parl.ai/docs/tasks.html)
 - [Creating a seq2seq agent](https://parl.ai/docs/tutorial_torch_generator_agent.html)
 - [List of available agents](https://parl.ai/docs/agents_list.html)
 - [Model zoo (list pretrained models)](https://parl.ai/docs/zoo.html)
 - [Running crowdsourcing tasks](http://parl.ai/docs/tutorial_crowdsourcing.html)
 - [Plug into Facebook Messenger](https://parl.ai/docs/tutorial_chat_service.html)


## Examples

A large set of scripts can be found in [`parlai/scripts`](https://github.com/facebookresearch/ParlAI/tree/main/parlai/scripts). Here are a few of them.
Note: If any of these examples fail, check the [installation section](#installing-parlai) to see if you have missed something.

Display 10 random examples from the SQuAD task
```bash
parlai display_data -t squad
```

Evaluate an IR baseline model on the validation set of the Personachat task:
```bash
parlai eval_model -m ir_baseline -t personachat -dt valid
```

Train a single layer transformer on PersonaChat (requires pytorch and torchtext).
Detail: embedding size 300, 4 attention heads,  2 epochs using batchsize 64, word vectors are initialized with fasttext and the other elements of the batch are used as negative during training.
```bash
parlai train_model -t personachat -m transformer/ranker -mf /tmp/model_tr6 --n-layers 1 --embedding-size 300 --ffn-size 600 --n-heads 4 --num-epochs 2 -veps 0.25 -bs 64 -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc --candidates batch
```

## Code Organization

The code is set up into several main directories:

- [**core**](https://github.com/facebookresearch/ParlAI/tree/main/parlai/core): contains the primary code for the framework
- [**agents**](https://github.com/facebookresearch/ParlAI/tree/main/parlai/agents): contains agents which can interact with the different tasks (e.g. machine learning models)
- [**scripts**](https://github.com/facebookresearch/ParlAI/tree/main/parlai/scripts): contains a number of useful scripts, like training, evaluating, interactive chatting, ...
- [**tasks**](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks): contains code for the different tasks available from within ParlAI
- [**mturk**](https://github.com/facebookresearch/ParlAI/tree/main/parlai/mturk): contains code for setting up Mechanical Turk, as well as sample MTurk tasks
- [**messenger**](https://github.com/facebookresearch/ParlAI/tree/main/parlai/chat_service/services/messenger): contains code for interfacing with Facebook Messenger
- [**utils**](https://github.com/facebookresearch/ParlAI/tree/main/parlai/utils): contains a wide number of frequently used utility methods
- [**crowdsourcing**](https://github.com/facebookresearch/ParlAI/tree/main/parlai/crowdsourcing): contains code for running crowdsourcing tasks, such as on Amazon Mechanical Turk
- [**chat_service**](https://github.com/facebookresearch/ParlAI/tree/main/parlai/chat_service/services/messenger): contains code for interfacing with services such as Facebook Messenger
- [**zoo**](https://github.com/facebookresearch/ParlAI/tree/main/parlai/zoo): contains code to directly download and use pretrained models from our model zoo

## Support
If you have any questions, bug reports or feature requests, please don't hesitate to post on our [Github Issues page](https://github.com/facebookresearch/ParlAI/issues).
You may also be interested in checking out our [FAQ](https://parl.ai/docs/faq.html) and
our [Tips n Tricks](https://parl.ai/docs/tutorial_tipsntricks.html).

Please remember to follow our [Code of Conduct](https://github.com/facebookresearch/ParlAI/blob/main/CODE_OF_CONDUCT.md).

## Contributing
We welcome PRs from the community!

You can find information about contributing to ParlAI in our
[Contributing](https://github.com/facebookresearch/ParlAI/blob/main/CONTRIBUTING.md)
document.


## The Team
ParlAI is currently maintained by Moya Chen, Emily Dinan, Dexter Ju, Mojtaba
Komeili, Spencer Poff, Pratik Ringshia, Stephen Roller, Kurt Shuster,
Eric Michael Smith, Megan Ung, Jack Urbanek, Jason Weston, Mary Williamson,
and Jing Xu. Kurt Shuster is the current Tech Lead.

Former major contributors and maintainers include Alexander H. Miller, Margaret
Li, Will Feng, Adam Fisch, Jiasen Lu, Antoine Bordes, Devi Parikh, Dhruv Batra,
Filipe de Avila Belbute Peres, Chao Pan, and Vedant Puri.

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
ParlAI is MIT licensed. See the **[LICENSE](https://github.com/facebookresearch/ParlAI/blob/main/LICENSE)** file for details.
