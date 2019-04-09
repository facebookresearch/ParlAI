#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Interact with a pre-trained retrieval model.
"""
from parlai.scripts.interactive import setup_args, interactive
from parlai.zoo.wizard_of_wikipedia\
    .full_dialogue_retrieval_model import download


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='projects:wizard_of_wikipedia:interactive_retrieval',
        retriever_model_file='models:wikipedia_full/tfidf_retriever/model',
        responder_model_file='models:wizard_of_wikipedia/'
                             'full_dialogue_retrieval_model/model',
    )
    opt = parser.parse_args(print_args=False)
    download(opt['datapath'])
    interactive(opt, print_parser=parser)
