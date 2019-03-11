#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Generates a pytorch data file from the training data; for use in the
PytorchDataTeacher.

Note that with our given implementation of batch act, episodes are compressed
such that each episode is one example for a model.

One can set the ``--context-len`` flag to specify how many past utterances
are used in a flattened episode.
"""
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.scripts.build_dict import build_dict, setup_args as dict_setup
import copy
import os
import json
import random
import collections
import torch
import tqdm
from collections import deque


def get_pyt_dict_file(opt):
    if opt.get('dict_file') and os.path.exists(opt.get('dict_file')):
        return opt['dict_file']
    if opt.get('dict_file') is None and opt.get('model_file'):
        return opt['model_file'] + '.dict'
    if not opt['pytorch_teacher_task']:
        opt['pytorch_teacher_task'] = opt['task']
    return os.path.join(
        opt.get('datapath', '.'),
        '{}_pyt_data'.format(opt['pytorch_teacher_task'].replace(':', '_')),
        opt['datatype'].split(':')[0],
        'dict')


def setup_args():
    from parlai.core.params import ParlaiParser
    parser = ParlaiParser(True, True, 'Builds a pytorch data file.')
    parser.add_pytorch_datateacher_args()
    return dict_setup(parser)


def make_serializable(obj):
    new_obj = {}
    for key, val in obj.items():
        if isinstance(val, (int, str, bytes, dict, list, tuple, bool)):
            new_obj[key] = val
        elif isinstance(val, collections.Mapping):
            new_obj[key] = dict(val)
        elif isinstance(val, collections.Sequence):
            new_obj[key] = list(val)
        elif torch.is_tensor(val):
            new_obj[key] = {'value': val.tolist(),
                            'deserialized_tensor': True,
                            'type': str(val.dtype)}
    return new_obj


def build_data(opt):
    if not opt.get('model', False):
        opt['model'] = 'repeat_label'
    preprocess = opt.get('pytorch_preprocess', True)
    opt['dict_file'] = get_pyt_dict_file(opt)
    dictionary = None
    if 'dict_maxexs' in opt:
        # Note: only build dictionary if dict loop args specified
        dictionary = build_dict(opt, skip_if_built=True)
    agent = create_agent(opt)
    # If build teacher not specified, we are simply looking for the file
    if not opt.get('pytorch_teacher_task', None):
        df = opt.get('pytorch_datapath')
        # check if the user set a datafile
        if not df:
            raise Exception('Tried to find data but `--pytorch-datapath` is not set')
        # check if the user provided the already built file
        if 'pytorch' not in df:
            df += '.pytorch' + (
                agent.getID() if opt.get('pytorch_preprocess', True) else ''
            )
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
    ordered_opt.pop('pytorch_teacher_dataset')
    ordered_opt['no_cuda'] = True
    world_data = create_task(ordered_opt, agent)
    teacher = world_data.agents[0]
    agent = world_data.agents[1]
    datapath = os.path.join(opt.get('datapath', '.'),
                            '{}_pyt_data'.format(
                                ordered_opt['task'].replace(':', '_')),
                            dt)
    if preprocess:
        datapath += '_{}_preprocess'.format(agent.getID().replace(':', '_'))
    if os.path.isdir(datapath) and 'data_length' in os.listdir(datapath):
        # Data already built
        print("[ pytorch data already built, at {}. ]".format(datapath))
        return datapath
    print(
        '----------\n[ setting up pytorch data, saving to {}/ ]\n----------'.format(
            datapath
        )
    )
    os.makedirs(datapath, exist_ok=True)
    num_eps = 0
    num_exs = 0
    current = []
    episode_done = False
    include_labels = opt.get('pytorch_include_labels', True)
    context_length = opt.get('pytorch_context_length', -1)
    context = deque(maxlen=context_length if context_length > 0 else None)
    total_exs = world_data.num_examples()
    pbar = tqdm.tqdm(
        total=total_exs, unit='ex', unit_scale=True,
        desc='Building pytorch data'
    )
    idx_to_char = []
    cumulative_char_len = 0
    # pass examples to dictionary
    with open(os.path.join(datapath, 'data'), 'w') as pytorch_data:
        while num_exs < total_exs:
            while not episode_done:
                action = teacher.act()
                current.append(action)
                episode_done = action.get('episode_done', False)

            # build separate episodes
            for ex in current:
                context.append(ex.get('text', ''))
                if len(context) > 1:
                    ex['text'] = '\n'.join(context)
                ex['episode_done'] = True
                labels = ex.get('labels', ex.get('eval_labels', None))
                if labels is not None and include_labels:
                    context.append(random.choice(labels))
                # generate observation from new example
                if preprocess:
                    ex = agent.observe(ex)
                    ex.pop('label_candidates', '')
                    ex['preprocessed'] = True
                num_eps += 1
                num_exs += 1
                pbar.update(1)
                ex_len = pytorch_data.write(json.dumps(make_serializable(ex)) + "\n")
                idx_to_char.append(cumulative_char_len)
                cumulative_char_len += ex_len
            # reset
            episode_done = False
            current.clear()
            context.clear()
    pbar.close()
    with open(os.path.join(datapath, 'char_index'), 'w') as char_index:
        json.dump(idx_to_char, char_index)
    with open(os.path.join(datapath, 'data_length'), 'w') as pytorch_data_len:
        pytorch_data_len.write(json.dumps({'num_eps': num_eps,
                                           'num_exs': num_exs}))
    if dictionary:
        dictionary.save(get_pyt_dict_file(opt), sort=True)

    print('[ pytorch data built. ]')
    return datapath


if __name__ == '__main__':
    build_data(setup_args().parse_args())
