#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This file contains a list of all the models in the model zoo, the path to
load them, agents & tasks associated (e.g. they were trained using) and a
description. Using the path you should be able to download and use the model
automatically, e.g.:
 python examples/interactive.py --model-file\
    "models:wikipedia_2016-12-21/tfidf_retriever/drqa_docs"
"""

model_list = [
    {
        "id": "drqa",
        "path": "models:drqa/squad/model",
        "agent": "drqa",
        "task": "squad",
        "description": "drqa reader trained on SQuAD",
        "result": (
            "{'exs': 10570, 'accuracy': 0.6886, 'f1': 0.7821, 'hits@1': 0.689, 'hits@5': 0.689, 'hits@10': 0.689, 'hits@100': 0.689, 'bleu': 0.1364, 'train_loss': 0}"  # noqa: E501
        ),
    },
    {
        "id": "wikipedia_2016-12-21",
        "path": "models:wikipedia_2016-12-21/tfidf_retriever/drqa_docs",
        "agent": "tfidf_retriever",
        "task": "wikipedia:full",
        "example": (
            "python -m parlai.scripts.interactive --model tfidf_retriever "
            "-mf models:wikipedia_2016-12-21/tfidf_retriever/drqa_docs"
        ),
        "result": (
            """
            Enter Your Message: Yann LeCun
            [candidate_scores]: [507.05804682 390.18244433 279.24033928 269.60377042 214.00140589]
            [SparseTfidfRetrieverAgent]:
            Deep learning (also known as deep structured learning, hierarchical learning or deep machine learning) is a branch of machine learning based on a set of algorithms that attempt to model high level abstractions in data. In a simple case, you could have two sets of neurons: ones that receive an input signal and ones that send an output signal. When the input layer receives an input it passes on a modified version of the input to the next layer. In a deep network, there are many layers between the input and output (and the layers are not made of neurons but it can help to think of it that way), allowing the algorithm to use multiple processing layers, composed of multiple linear and non-linear transformations.

            Deep learning is part of a broader family of machine learning methods based on ...
            to commonsense reasoning which operates on concepts in terms of production rules of the grammar, and is a basic goal of both human language acquisition and AI. (See also Grammar induction.)
            """  # noqa: E501
        ),
        "description": (
            "Retrieval over Wikipedia dump, used for DrQA on the open squad "
            "dataset. This is the dump from the original paper, used for "
            "replicating results."
        )
    },
    {
        "id": "wikipedia_full",
        "path": "models:wikipedia_full/tfidf_retriever/model",
        "agent": "tfidf_retriever",
        "task": "wikipedia:full",
        "description": (
            "Retrieval over Wikipedia dump, used for DrQA on the open squad "
            "dataset."
        ),
        "example": "python -m parlai.scripts.interactive --model tfidf_retriever -mf models:wikipedia_full/tfidf_retriever/model",  # noqa: E501
        "result": (
            """
            Enter Your Message: Yann LeCun
            [candidate_scores]: [454.74038503 353.88863708 307.31353203 280.4501096  269.89960432]
            [SparseTfidfRetrieverAgent]:
            Yann LeCun (; born 1960) is a computer scientist with contributions in machine learning, computer vision, mobile robotics and computational neuroscience. He is well known for his work on optical character recognition and computer vision using convolutional neural networks (CNN), and is a founding father of convolutional nets. He is also one of the main creators of the DjVu image compression technology (together with Léon Bottou and Patrick Haffner). He co-developed the Lush programming language with Léon Bottou.

            Yann LeCun was born near Paris, France, in 1960. He received a Diplôme d'Ingénieur from the Ecole Superieure d'Ingénieur en Electrotechnique et Electronique (ESIEE), Paris in 1983, and a PhD in Computer Science from Université Pierre et Marie Curie in 1987 during which he ...
            of Science and Technology in Saudi Arabia because he was considered a terrorist in the country in view of his atheism.

            In 2018 Yann LeCun picked a fight with a robot to support Facebook AI goals.
            """  # noqa: E501
        ),
    },
    {
        "id": "twitter",
        "path": "models:twitter/seq2seq/twitter_seq2seq_model",
        "agent": "legacy:seq2seq:0",
        "task": "twitter",
        "description": (
            "Generic conversational model trained on the twitter task"
        ),
        "result": "{'exs': 10405, 'accuracy': 0.001538, 'f1': 0.07537, 'bleu': 0.002304, 'loss': 3.93, 'ppl': 50.9}",  # noqa: E501
    }
]
