# Reducing Hallucination in Conversational Agents

### _Retrieval Augmentation Reduces Hallucination in Conversation_

Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela\*, Jason Weston\*

\* Equal Contribution.

## Abstract

Despite showing increasingly human-like conversational abilities, state-of-the-art dialogue models often suffer from factual incorrectness and hallucination of knowledge (Roller et al., 2020). In this work we explore the use of neural-retrieval-in-the-loop architectures - recently shown to be effective in open-domain QA (Lewis et al., 2020b; Izacard and Grave, 2020) - for knowledge-grounded dialogue, a task that is arguably more challenging as it requires querying based on complex multi-turn dialogue context and generating conversationally coherent responses. We study various types of architectures with multiple components - retrievers, rankers, and encoder-decoders - with the goal of maximizing knowledgeability while retaining conversational ability. We demonstrate that our best models obtain state-of-the-art performance on two knowledge-grounded conversational tasks. The models exhibit open-domain conversational capabilities, generalize effectively to scenarios not within the training data, and, as verified by human evaluations, substantially reduce the well-known problem of knowledge hallucination in state-of-the-art chatbots.

## Paper

[Link to arXiv](https://arxiv.org/abs/2104.07567)

### Model Evaluations


<p align="center"><img width="85%" src="Human_Evals.png" /></p>

### Example Model Outputs

<p align="center"><img width="85%" src="Model_Outputs.png" /></p>

## Tasks

### Wizard of Wikipedia (WoW)

You can access the [WoW](https://openreview.net/forum?id=r1l73iRqKm) dataset in ParlAI via the following:

    parlai dd -t wizard_of_wikipedia

### CMU Document Grounded Conversations (CMU_DoG)

You can access the [CMU_DoG](https://arxiv.org/abs/1809.07358) dataset in ParlAI via the following:

    parlai dd -t cmu_dog

To use the modified splits as described in the [paper](https://arxiv.org/abs/2104.07567), set the following flags for the seen/unseen splits, respectively:

    parlai dd -t cmu_dog --cmu-dog-split-type seen

    parlai dd -t cmu_dog --cmu-dog-split-type unseen --datatype test

## Pre-Trained Models

Coming Soon!!

## Train your Own Models

Coming Soon!!
