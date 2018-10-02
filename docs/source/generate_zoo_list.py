#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.zoo.model_list import model_list

fout = open('zoo_list.inc', 'w')

for model in model_list:
    name = model['id'].replace('_', ' ').title()
    fout.write(name)
    fout.write('\n')
    fout.write('-' * len(name))
    fout.write('\n\n')

    fout.write(model['description'])
    fout.write('\n\n')

    fout.write('Example invocation:\n')
    fout.write(
        '``python -m parlai.scripts.eval_model -a {} -t {} -mf {}``\n'
        .format(
            model['agent'],
            model['task'],
            model['path'],
        )
    )
    fout.write('\n')

fout.close()
