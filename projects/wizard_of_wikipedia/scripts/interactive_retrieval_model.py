#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Interact with a pre-trained retrieval model.
"""
from parlai.scripts.interactive import setup_args, interactive
from projects.wizard_of_wikipedia.wizard_transformer_ranker.wizard_transformer_ranker import (
    WizardTransformerRankerAgent,
)


if __name__ == '__main__':
    parser = setup_args()
    WizardTransformerRankerAgent.add_cmdline_args(parser)
    parser.set_params(
        task='wizard_of_wikipedia',
        model='projects:wizard_of_wikipedia:wizard_transformer_ranker',
        model_file='models:wizard_of_wikipedia/full_dialogue_retrieval_model/model',
        datatype='test',
        n_heads=6,
        ffn_size=1200,
        embeddings_scale=False,
        delimiter=' __SOC__ ',
        n_positions=1000,
        legacy=True,
        eval_candidates='fixed',
        interactive_mode=True,
    )
    opt = parser.parse_args(print_args=False)
    interactive(opt, print_parser=parser)
