# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""This file contains a list of all the models in the model zoo,
the path to load them, agents & tasks associated (e.g. they were trained using) and a description.
Using the path you should be able to download and use the model automatically, e.g.:
 python examples/interactive.lua --model-file "models:wikipedia_2016-12-21/tfidf_retriever/drqa_docs"
"""

model_list = [
    {
        "id": "wikipedia_2016-12-21_retriever",
        "path": "models:wikipedia_2016-12-21/tfidf_retriever/drqa_docs",
        "agent":  "tfidf_retriever",
        "task": "wikipedia_2016-12-21",
        "description": "retrieval over Wikipedia dump, used for DrQA on the open squad dataset. This is the dump from the original paper, used for replicating results."
    },
    {
        "id": "wikipedia_full_retriever",
        "path": "models:wikipedia_full/tfidf_retriever/model",
        "agent":  "tfidf_retriever",
        "task": "wikipedia_full",
        "description": "retrieval over Wikipedia dump, used for DrQA on the open squad dataset"
    },
    {
        "id": "drqa_squad_reader",
        "path": "models:drqa/squad/model",
        "agent":  "drqa",
        "task": "squad",
        "description": "drqa reader trained on SQuAD"
    },
]
