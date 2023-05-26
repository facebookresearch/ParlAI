# Multi-Party Chat: Conversational Agents in Group Settings with Humans and Models

Jimmy Wei, Kurt Shuster, Arthur Szlam, Jason Weston, Jack Urbanek, Mojtaba Komeili

## Abstract

Current dialogue research primarily studies pairwise (two-party) conversations, and does not address the everyday setting where more than two speakers converse together. In this work, we both collect and evaluate multi-party conversations to study this more general case. We use the LIGHT environment to construct grounded conversations, where each participant has an assigned character to role-play. We thus evaluate the ability of language models to act as one or more characters in such conversations. Models require two skills that pairwise-trained models appear to lack: (1) being able to decide when to talk; (2) producing coherent utterances grounded on multiple characters. We compare models trained on our new dataset to existing pairwise-trained dialogue models, as well as large language models with few-shot prompting. We find that our new dataset, MultiLIGHT, which we will publicly release, can help bring significant improvements in the group setting.

## Paper

[Link](https://arxiv.org/abs/2304.13835)


## Data

We designed a new crowdsourcing task ([link](https://github.com/facebookresearch/LIGHT/tree/main/crowdsourcing/dialogues/multi_party_chat)) and collected a new dataset.
Data is available via a ParlAI style teacher in [LIGHT](https://github.com/facebookresearch/LIGHT).
```.sh
light dd -t light:multilight \
--add-location-to-context true \
--add-personas-to-context true
```
See [this](https://github.com/facebookresearch/LIGHT/tree/main/light/modeling/tasks/multilight
) for more details on available teachers and their flags.

Rendered sample data:
<p align="center"><img width="50%" src="DatasetExample.png" /></p>

## Models

We released three models from this project.
These are the our best performing models that were included in our human evaulation.

* `utterance_3B`: the best performing *Utterance only* model trained on LIGHT, LIGHT Wild and MultiLIGHT, multi-tasked (3B parameters size).
* `utterance_400m`: the best performing *Utterance only* model trained on LIGHT, LIGHT Wild and MultiLIGHT, multi-tasked (400m parameters size).
* `speaker`: predicts the next speaker (based on BART-large).

### Running models

You can run these models with the existing ParlAI dataset, for example
```.sh
parlai eval_model -mf zoo:multilight/utterance_3B/model --task wizard_of_internet
```

> *Note*: Due match the data distribution (location and persona descriptions, and their keywords, character names etc., you may not get high performance).

The main dataset that were are fine-tuned on is available in LIGHT.
You are able to run these models with commands templates identical to ParlAI, for example:

```.sh
light eval_model -mf zoo:multilight/utterance_3B/model -t light:multilight \
--add-location-to-context true --add-personas-to-context true \
--include-speaker-in-label false --add-speaker-to-context-end true
```
