#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.scripts.train_model import setup_args, TrainLoop

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='wizard_of_wikipedia:generator:random_split',
        model='projects.wizard_of_wikipedia.generator.agents:EndToEndAgent',
        model_file='/tmp/end2end_generator/model',
        dict_lower=True,
        dict_tokenizer='bpe',
        n_layers=5,
        n_heads=2,
        dropout=0.20,
        ffn_size=512,
        embedding_size=256,
        log_every_n_secs=10,
        validation_patience=12,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_every_n_epochs=0.5,
        n_positions=128,
        truncate=128,
        max_knowledge=32,
        knowledge_alpha=0.95,
        knowledge_truncate=32,
        learningrate=5e-4,
        warmup_updates=5000,
        clip=0.1,
        lr_scheduler='invsqrt',
        embedding_type='fasttext',
        beam_size=1,
        skip_generation=False,
        batchsize=64,
    )
    TrainLoop(parser.parse_args()).train()
