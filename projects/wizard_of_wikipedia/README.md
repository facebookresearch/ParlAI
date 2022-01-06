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
See the [ParlAI quickstart for help](http://www.parl.ai/docs/tutorial_quick.html).

The ParlAI MTurk collection scripts are also
[available](https://github.com/facebookresearch/ParlAI/tree/main/parlai/mturk/README.md) in an older release of ParlAI (see the `wizard_of_wikipedia` task),
for those interested in replication, analysis, or additional data collection.
The MTurk task for evaluating pre-trained models is made available in this
directory.

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
End-to-end Transformer MemNet        | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 63.5 | 97.3
Two-Stage Transformer Memnet         | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 46.5 | 84.8
Vanilla Transformer (no knowledge)   | [Dinan et al. (2019)](https://arxiv.org/abs/1811.01241) | 41.8 | 87.0


## Viewing data

You can view the standard training set with:

    parlai display_data -t wizard_of_wikipedia -dt train

The knowledge returned from a standard IR system appears in the knowledge field (but you can also use your own knowledge system, accessing Wikipedia yourself, we use the dump in "-t wikipedia".
The field 'checked_sentence' indicates the gold knowledge the annotator labeled.

## Pretrained models

## End-to-End generative

You can evaluate the pretrained End-to-end generative models via:

    parlai eval_model \
        -bs 64 -t wizard_of_wikipedia:generator:random_split \
        -mf models:wizard_of_wikipedia/end2end_generator/model


This produces the following metrics:

    {'f1': 0.1717, 'ppl': 61.21, 'know_acc': 0.2201, 'know_chance': 0.02625}

This differs slightly from the results in the paper, as it is a recreation trained
from scratch for public release.

You can also evaluate the model on the unseen topic split too:

    parlai eval_model \
        -bs 64 -t wizard_of_wikipedia:generator:topic_split \
        -mf models:wizard_of_wikipedia/end2end_generator/model

This will produce:

    {'f1': 0.1498, 'ppl': 103.1, 'know_acc': 0.1123, 'know_chance': 0.02496}

You can also interact with the model with:

    parlai interactive -mf models:wizard_of_wikipedia/end2end_generator/model -t wizard_of_wikipedia

_Note_: an unofficial Tensorflow implementation of the End2End generative model can be found [here](https://lucehe.github.io/wow/).

## Retrieval Model

You can evaluate a retrieval model on the full dialogue task by running the
following script:

    python projects/wizard_of_wikipedia/scripts/eval_retrieval_model.py

You can run an interactive session with the model with:

    python projects/wizard_of_wikipedia/scripts/interactive_retrieval_model.py

Check back later for more pretrained models soon!

## Raw Data

If you want to work with the raw data, we describe the setup of the `json` files.

Each `<train/val/test>.json` file is a list of all dialogues in that split. An entry in the list has the following keys:

- `chosen_topic`: the chosen topic of the conversation
- `persona`: a corresponding persona that motivated the topic (note this was not used during data collection)
- `wizard_eval`: an evaluation of the wizard provided by the apprentice at the end of the dialogue
- `dialog`: the list of dialogue turns
- `chosen_topic_passage`: a list of sentences from the wiki passage corresponding to the chosen topic

The entries of `dialog` (may) have the following keys; some are omitted for the apprentice:

- `speaker`: either `"wizard"` or `"apprentice"`
- `text`: what the speaker wrote
- `retrieved_topics`: the topics retrieved for that utterance
- `retrieved_passages`: a list of 1 entry dicts, mapping a topic to the sentences in the passage
- `checked_sentence`: (wizard only) a 1 entry dict mapping the topic to the chosen sentence by the wizard
- `checked_passage`: (wizard only) a 1 entry dict mapping the topic to the chosen topic by the wizard


## Citation

If you use the dataset or models in your own work, please cite with the
following BibTex entry:

    @inproceedings{dinan2019wizard,
      author={Emily Dinan and Stephen Roller and Kurt Shuster and Angela Fan and Michael Auli and Jason Weston},
      title={{W}izard of {W}ikipedia: Knowledge-powered Conversational Agents},
      booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2019},
    }
