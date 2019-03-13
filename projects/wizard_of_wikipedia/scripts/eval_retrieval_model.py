#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.build_data import download_models
from parlai.core.params import ParlaiParser
from parlai.scripts.eval_model import eval_model
from projects.wizard_of_wikipedia.wizard_transformer_ranker\
    .wizard_transformer_ranker import WizardTransformerRankerAgent

"""Evaluate pre-trained retrieval model on the full Wizard Dialogue task.

NOTE: Metrics here differ slightly to those reported in the paper as a result
of code changes.
"""

if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument('-n', '--num-examples', default=100000000)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    WizardTransformerRankerAgent.add_cmdline_args(parser)
    parser.set_defaults(
        task='wizard_of_wikipedia',
        model='projects:wizard_of_wikipedia:wizard_transformer_ranker',
        model_file='models:wizard_of_wikipedia/retrieval_models/model',
        datatype='test',
        n_heads=6,
        ffn_size=1200,
        embeddings_scale=False,
        delimiter=' __SOC__ ',
        n_positions=1000,
        legacy=True
    )

    opt = parser.parse_args()
    # download models
    fnames = ['full_dialogue_retrieval_model.tgz']
    opt['model_type'] = 'retrieval_models'
    download_models(opt, fnames, 'wizard_of_wikipedia')

    eval_model(parser)
