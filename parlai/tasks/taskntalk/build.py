#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import os
import random

import parlai.core.build_data as build_data
from parlai.utils.io import PathManager


def build(opt):
    """
    Create train and validation data for synthetic shapes described by attributes.
    """
    dpath = os.path.join(opt['datapath'], 'taskntalk')

    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')
        build_data.make_dir(os.path.join(dpath, 'large'))
        build_data.make_dir(os.path.join(dpath, 'small'))

        # save training and validation data
        to_save = {
            'attributes': ['color', 'shape', 'style'],
            'task_defn': [
                ['color', 'shape'],
                ['shape', 'color'],
                ['color', 'style'],
                ['style', 'color'],
                ['shape', 'style'],
                ['style', 'shape'],
            ],
        }
        split_data = {}

        # small dataset properties
        properties = {
            'color': ['red', 'green', 'blue', 'purple'],
            'shape': ['square', 'triangle', 'circle', 'star'],
            'style': ['dotted', 'solid', 'filled', 'dashed'],
        }
        to_save['properties'] = properties
        # properties.values() not used directly to maintain order
        data_verbose = list(
            itertools.product(*[properties[key] for key in to_save['attributes']])
        )

        # randomly select train and rest of it is valid
        split_data['valid'] = random.sample(data_verbose, int(0.2 * len(data_verbose)))
        split_data['train'] = [s for s in data_verbose if s not in split_data['valid']]

        to_save['data'] = split_data['train']
        with PathManager.open(
            os.path.join(dpath, 'small', 'train.json'), 'w'
        ) as outfile:
            json.dump(
                to_save, outfile, indent=4, separators=(',', ': '), sort_keys=True
            )

        to_save['data'] = split_data['valid']
        with PathManager.open(
            os.path.join(dpath, 'small', 'valid.json'), 'w'
        ) as outfile:
            json.dump(
                to_save, outfile, indent=4, separators=(',', ': '), sort_keys=True
            )

        # large dataset properties
        properties = {
            'color': [
                'red',
                'green',
                'blue',
                'purple',
                'yellow',
                'cyan',
                'orange',
                'teal',
            ],
            'shape': [
                'square',
                'triangle',
                'circle',
                'star',
                'heart',
                'spade',
                'club',
                'diamond',
            ],
            'style': [
                'dotted',
                'solid',
                'filled',
                'dashed',
                'hstripe',
                'vstripe',
                'hgrad',
                'vgrad',
            ],
        }
        to_save['properties'] = properties
        data_verbose = list(
            itertools.product(*[properties[key] for key in to_save['attributes']])
        )
        split_data['valid'] = random.sample(data_verbose, int(0.8 * len(data_verbose)))
        split_data['train'] = [s for s in data_verbose if s not in split_data['valid']]

        to_save['data'] = split_data['train']
        with PathManager.open(
            os.path.join(dpath, 'large', 'train.json'), 'w'
        ) as outfile:
            json.dump(
                to_save, outfile, indent=4, separators=(',', ': '), sort_keys=True
            )

        to_save['data'] = split_data['valid']
        with PathManager.open(
            os.path.join(dpath, 'large', 'valid.json'), 'w'
        ) as outfile:
            json.dump(
                to_save, outfile, indent=4, separators=(',', ': '), sort_keys=True
            )

        # Mark the data as built.
        build_data.mark_done(dpath)
