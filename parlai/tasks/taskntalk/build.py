# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import itertools
import json
import os
import random

import parlai.core.build_data as build_data


def build(opt):
    """Create train and validation data for synthetic shapes described by attributes."""
    dpath = os.path.join(opt['datapath'], 'shapes_small')

    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        # save training and validation data
        to_save = {
            'attributes': ['color', 'shape', 'style'],
            'task_defn': [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
        }
        split_data = {}

        # small dataset properties
        properties = {
            'color': ['red', 'green', 'blue', 'purple'],
            'shape': ['square', 'triangle', 'circle', 'star'],
            'style': ['dotted', 'solid', 'filled', 'dashed']
        }
        to_save['properties'] = properties
        # properties.values() not used directly to maintain order
        data_verbose = list(itertools.product(*[properties[key] for key in to_save['attributes']]))

        # randomly select train and rest of it is valid
        split_data['valid'] = random.sample(data_verbose, int(0.8 * len(data_verbose)))
        split_data['train'] = [s for s in data_verbose if s not in split_data['val']]

        to_save['data'] = split_data['train']
        with open(os.path.join(dpath, 'shapes_small', 'train.json'), 'w') as outfile:
            json.dump(to_save, outfile, indent=4, separators=(',', ': '), sort_keys=True)

        to_save['data'] = split_data['valid']
        with open(os.path.join(dpath, 'shapes_small', 'valid.json'), 'w') as outfile:
            json.dump(to_save, outfile, indent=4, separators=(',', ': '), sort_keys=True)

        # large dataset properties
        properties = {
            'color': ['red', 'green', 'blue', 'purple', 'yellow', 'cyan', 'orange', 'teal'],
            'shape': ['square', 'triangle', 'circle', 'star', 'heart', 'pentagon', 'hexagon', 'ring'],
            'style': ['dotted', 'solid', 'filled', 'dashed', 'hstripe', 'vstripe', 'hgrad', 'vgrad']
        }
        to_save['properties'] = properties
        data_verbose = list(itertools.product(*[properties[key] for key in to_save['attributes']]))
        split_data['valid'] = random.sample(data_verbose, int(0.8 * len(data_verbose)))
        split_data['train'] = [s for s in data_verbose if s not in split_data['val']]

        to_save['data'] = split_data['train']
        with open(os.path.join(dpath, 'shapes_large', 'train.json'), 'w') as outfile:
            json.dump(to_save, outfile, indent=4, separators=(',', ': '), sort_keys=True)

        to_save['data'] = split_data['valid']
        with open(os.path.join(dpath, 'shapes_large', 'valid.json'), 'w') as outfile:
            json.dump(to_save, outfile, indent=4, separators=(',', ': '), sort_keys=True)

        # Mark the data as built.
        build_data.mark_done(dpath)
