# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


from parlai_internal.projects.param_sweep_utils.param_sweep import run_grid

SWEEP_NAME = 'transformer_books'
NUM_GPUS = 1

name_keys = {}

grid = {
    '-pyt': [
        # 'wizard_of_wikipedia:WizardDialogKnowledge',
        # 'internal:Reddit:redditPytorchData'
        'internal:toronto_books'
    ],
    '--model': [
        'projects.wizard.wizard_transformer_ranker.wizard_transformer_ranker:WizardTransformerRankerAgent',
    ],
    '--dict-file': [
        '/private/home/edinan/ParlAI/data/toronto_books/toronto_books.dict'
    ],
    '-dt': [
        'train:stream'
    ],
    '--n-layers': [
        4
    ],
    '--lr-factor': [
        0.5,
        1,
    ],
    '--n-heads': [
        6
    ],
    '-lr': [
        #0.0008,
        0.00008,
    ],
    '-opt': [
        'adamax',
    ],
    '-bs': [
        32,
        #64,
    ],
    '--init-model': [
        '/checkpoint/edinan/20190111/transformer_books/lr-factor=0.5_lr=8e-05_bs=32/model'
    ],
    '--truncate': [
        128,
    ],
    '-vtim': [
        3600,
    ],
    '-vp': [
        15,
    ],
    '-stim': [
        60,
    ],
    '-vme': [
        10000
    ],
    '--validation-metric': [
        'accuracy'
    ],
    '--validation-metric-mode': [
        'max'
    ],
    '--save-after-valid': [
        True
    ],
}

if __name__ == '__main__':
    run_grid(grid, name_keys, SWEEP_NAME, partition='dev',
             jobtime='48:00:00', prefix='python -u examples/train_model.py',
             gpus=NUM_GPUS, create_model_file=True)
