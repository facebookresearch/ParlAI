#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.scripts.interactive import setup_args, interactive

import random

if __name__ == '__main__':
    random.seed(42)
    parser = setup_args()

    parser.set_params(batchsize=1, beam_size=20, beam_min_n_best=10)

    print('\n' + '*' * 80)
    print('WARNING: This dialogue model is a research project that was trained on a')
    print(
        'large amount of open-domain Twitter data. It may generate offensive content.'
    )
    print('*' * 80 + '\n')

    interactive(parser.parse_args())
