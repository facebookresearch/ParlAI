#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser
from parlai.utils.io import PathManager


def setup_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-if', '--infile', type=str, default='data/ConvAI2/train_self_original.txt'
    )
    parser.add_argument(
        '-of', '--outfile', type=str, default='projects/metadialog/convai2_cands.txt'
    )
    config = vars(parser.parse_args())
    return config


def main(config):
    with PathManager.open(config['infile'], 'r') as fin, PathManager.open(
        config['outfile'], 'w'
    ) as fout:
        for line in fin.readlines():
            if 'persona' in line:
                continue
            first_space = line.index(' ')
            first_tab = line.index('\t')
            candidate = line[first_space + 1 : first_tab]
            fout.write(candidate + '\n')


if __name__ == '__main__':
    config = setup_args()
    main(config)
