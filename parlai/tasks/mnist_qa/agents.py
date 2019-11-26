#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a simple question answering task on the MNIST dataset. In each episode, agents
are presented with a number, which they are asked to identify.

Useful for debugging and checking that one's image model is up and running.
"""

from parlai.core.teachers import DialogTeacher
from .build import build

import json
import os


def _path(opt):
    build(opt)
    dt = opt['datatype'].split(':')[0]

    labels_path = os.path.join(opt['datapath'], 'mnist', dt, 'labels.json')
    image_path = os.path.join(opt['datapath'], 'mnist', dt)
    return labels_path, image_path


class MnistQATeacher(DialogTeacher):
    """
    This version of MNIST inherits from the core Dialog Teacher, which just requires it
    to define an iterator over its data `setup_data` in order to inherit basic metrics,
    a `act` function, and enables Hogwild training with shared memory with no extra
    work.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype'].split(':')[0]
        labels_path, self.image_path = _path(opt)
        opt['datafile'] = labels_path
        self.id = 'mnist_qa'
        self.num_strs = [
            'zero',
            'one',
            'two',
            'three',
            'four',
            'five',
            'six',
            'seven',
            'eight',
            'nine',
        ]

        super().__init__(opt, shared)

    def label_candidates(self):
        return [str(x) for x in range(10)] + self.num_strs

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as labels_file:
            self.labels = json.load(labels_file)

        self.question = 'Which number is in the image?'
        episode_done = True

        for i in range(len(self.labels)):
            img_path = os.path.join(self.image_path, '%05d.bmp' % i)
            label = [self.labels[i], self.num_strs[int(self.labels[i])]]
            yield (self.question, label, None, None, img_path), episode_done


class DefaultTeacher(MnistQATeacher):
    pass
