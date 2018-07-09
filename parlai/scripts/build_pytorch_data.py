# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Generates a pytorch data file from the training data; for use in the
PytorchDataTeacher.

Note that with our given implementation of batch act, episodes are compressed
such that each episode is one example for a model.

One can set the `--context-len` flag to specify how many past utterances
are used in a flattened episode.
"""
from parlai.core.agents import create_agent, create_task_agent_from_taskname
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.core.utils import ProgressLogger
import copy
import os
import json
import random
import collections
import torch
from collections import deque

def setup_args():
    return ParlaiParser(True, True)

def make_serializable(obj):
    new_obj = {}
    for key, val in obj.items():
        if isinstance(val, (int, str, bytes, dict, list, tuple, bool)):
            new_obj[key] = val
        elif isinstance(val, collections.Mapping):
            new_obj[key] = dict(val)
        elif isinstance(val, collections.Sequence):
            new_obj[key] = list(val)
        elif isinstance(val, torch.Tensor):
            new_obj[key] = val.tolist()
    return new_obj


def build_data(opt):
    if not opt.get('model', False):
        opt['model'] = 'repeat_label'
    agent = create_agent(opt)
    #If build teacher not specified, we are simply looking for the file
    if not opt.get('pytorch_teacher_task', None):
        df = opt.get('pytorch_datafile')
        # check if the user set a datafile
        if not df:
            raise Exception('Tried to find data but `--pytorch-datafile` is not set')
        # check if the user provided the already built file
        if 'pytorch' not in df:
            df += '.pytorch' + (agent.getID() if opt.get('pytorch_preprocess', True) else '')
        if not os.path.isfile(df):
            raise Exception('Tried to find data but it is not built, please'
                            'specify `--pytorch-teacher-task`')
        else:
            return df

    ordered_opt = copy.deepcopy(opt)
    # we use streaming to build the data
    dt = opt['datatype'].split(':')[0]
    ordered_opt['datatype'] = dt + ':ordered:stream'
    ordered_opt['numthreads'] = 1
    ordered_opt['batchsize'] = 1
    ordered_opt['task'] = ordered_opt['pytorch_teacher_task']
    ordered_opt['no_cuda'] = True
    world_data = create_task(ordered_opt, agent)
    teacher = world_data.agents[0]
    agent = world_data.agents[1]

    datafile = teacher.datafile if hasattr(teacher, 'datafile') else opt.get('pytorch_datafile')
    if not datafile:
        raise Exception('Tried to build data but either `pytorch-teacher-task` does not '
                        'have a datafile or `--pytorch-datafile` is not set')

    if isinstance(datafile, collections.Sequence) and not type(datafile) == str:
        datafile = datafile[0] + "".join(["_".join(d.split("/")) for d in datafile[1:]])
    pytorch_datafile = datafile + ".pytorch"
    preprocess = opt.get('pytorch_preprocess', True)
    if preprocess:
        pytorch_datafile += agent.getID()
    if os.path.isfile(pytorch_datafile):
        # Data already built
        print("[ pytorch data already built. ]")
        return pytorch_datafile
    print('----------\n[ setting up pytorch data, saving to {}. ]\n----------'.format(pytorch_datafile))

    num_eps = 0
    num_exs = 0
    current = []
    episode_done = False
    include_labels = opt.get('include_labels', True)
    context_length = opt.get('context_length', -1)
    context = deque(maxlen=context_length if context_length > 0 else None)
    logger = ProgressLogger(should_humanize=False, throttle=0.1)
    total_exs = world_data.num_examples()
    # pass examples to dictionary
    with open(pytorch_datafile, 'w') as pytorch_data:
        while num_exs < total_exs:
            while not episode_done:
                action = teacher.act()
                current.append(action)
                episode_done = action.get('episode_done', False)

            #build separate episodes
            for ex in current:
                context.append(ex.get('text', ''))
                if len(context) > 1:
                    ex['text'] = '\n'.join(context)
                ex['episode_done'] = True
                labels = ex.get('labels', ex.get('eval_labels', None))
                if labels is not None and include_labels:
                    context.append(random.choice(labels))
                #generate observation from new example
                if preprocess:
                    ex = agent.observe(ex)
                    ex.pop('label_candidates', '')
                    ex['preprocessed'] = True
                num_eps += 1
                num_exs += 1
                logger.log(num_exs, total_exs)
                pytorch_data.write(json.dumps(make_serializable(ex)) + "\n")
            #reset
            episode_done = False
            current.clear()
            context.clear()

    with open(pytorch_datafile + '.length', 'w') as pytorch_data_len:
        pytorch_data_len.write(json.dumps({'num_eps':num_eps, 'num_exs':num_exs}))

    print('[ pytorch data built. ]')
    return pytorch_datafile


if __name__ == '__main__':
    build_data(setup_args().parse_args())
