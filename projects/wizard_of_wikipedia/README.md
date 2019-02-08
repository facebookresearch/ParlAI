# ![mage](mage.png) Wizard of Wikipedia: Knowledge-Powered Conversational Agents

<p align="center"><img width="15%" src="parrot.png" /></p>

The Wizard of Wikipedia is an open-domain dialogue task for training agents
that can converse knowledgably about open-domain topics!
A detailed description may
be found in [Dinan et al. (ICLR 2019)](https://arxiv.org/abs/1811.01241).

## Abstract

In open-domain dialogue intelligent agents should exhibit the use of knowledge,
however there are few convincing demonstrations of this to date.  The most
popular sequence to sequence models typically "generate and hope"  generic
utterances  that can be memorized in the weights of the model when mapping from
input utterance(s) to output, rather than employing recalled knowledge as
context.  Use of knowledge has so far proved difficult, in part because of the
lack of a supervised learning benchmark task which exhibits knowledgeable open
dialogue with clear grounding.  To that end we collect and release a large
dataset with conversations  directly grounded with knowledge retrieved from
Wikipedia.  We then design architectures capable of retrieving knowledge,
reading and conditioning on it, and finally generating natural responses.  Our
best performing dialogue models are able to conduct knowledgeable discussions
on open-domain topics as evaluated by automatic metrics and human evaluations,
while our new benchmark allows for measuring further improvements in this
important research direction.

## Datasets

You can train your own ParlAI agent on the Wizard of Wikipedia task with
`-t wizard_of_wikipedia`.

The ParlAI MTurk collection scripts are also
[made available](https://github.com/facebookresearch/ParlAI/tree/master/parlai/mturk/tasks/wizard_of_wikipedia),
for those interested in replication, analysis, or additional data collection

## Leaderboard

### Human Evaluations
Model                                | Paper          | Seen Rating   | Unseen Rating
------------------------------------ | -------------- | ------------- | ---------------
Retrieval Trans MemNet               | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 3.43 | 3.14
Two-stage Generative Trans MemNet    | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 2.92 | 2.93
Human performance                    | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 4.13 | 4.34

### Retrieval Models

Model                                | Paper          | Test Seen R@1 | Test Unseen R@1
------------------------------------ | -------------- | ------------- | ---------------
Transformer MemNet (w/ pretraining)  | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 87.4 | 69.8
BoW Memnet                           | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 71.3 | 33.1
IR baseline                          | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 17.8 | 14.2
Random                               | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) |  1.0 |  1.0

### Generative Models

Model                                | Paper          | Test Seen PPL | Test Unseen PPL
------------------------------------ | -------------- | ------------- | ---------------
Two-Stage Transformer Memnet         | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 46.5 | 84.8
End-to-end Transformer MemNet        | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 63.5 | 97.3
Vanilla Transformer (no knowledge)   | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 41.8 | 87.0


## Pretrained Models

Finalized models are not yet released. Please check back here in the future.

## Citation

If you use the dataset or models in your own work, please cite with the
following BibTex entry:

    @inproceedings{dinan2019wizard,
      author={Emily Dinan and Stephen Roller and Kurt Shuster and Angela Fan and Michael Auli and Jason Weston},
      title={{W}izard of {W}ikipedia: Knowledge-powered Conversational Agents},
      booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2019},
    }

