#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.zoo.model_list import model_list

fout = open('zoo_list.inc', 'w')

for model in model_list:
    name = model.get('title').title()
    fout.write(name)
    fout.write('\n')
    fout.write('-' * len(name))
    fout.write('\n\n')

    fout.write(model['description'])
    fout.write('\n\n')

    fout.write('Example invocation:\n\n')
    if 'example' in model:
        example = model['example']
    else:
        example = (
            "python -m parlai.scripts.eval_model --model {} --task {} -mf {}"
            .format(model['agent'], model['task'], model['path'])
        )
    result = model.get('result', '').strip().split("\n")
    # strip leading whitespace from results
    result = [r.strip() for r in result]
    # make sure we indent for markdown though
    result = ["   " + r for r in result]
    result = "\n".join(result)
    fout.write('.. code-block:: \n\n   {}\n   ...\n{}\n\n'.format(example, result))
    fout.write('\n')

fout.close()
