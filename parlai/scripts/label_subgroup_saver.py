#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script for labeling a dataset with the specified classifier (works for other classifier-
based datasets)
"""

from parlai.core.script import ParlaiScript, register_script
from parlai.core.params import ParlaiParser
import parlai.utils.logging as logging
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
import json
from collections import defaultdict


def get_json_line(dic):
    json_list = []
    for key, val in dic.items():
        json_list.append({'id': key, 'text': val})
    return json.dumps({"dialog": [json_list]})


def get_txt_line(dic):
    line = [key + ':' + value for key, value in dic.items()]
    return '\t'.join(line) + '\tepisode_done:True'


ext_func = {'.json': get_json_line, '.txt': get_txt_line}


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(
            True,
            True,
            'Evaluate a dataset into subgroups and save datasets with subgroup_label',
        )
    parser.add_argument(
        '--save-loc',
        '-saveloc',
        type=str,
        default=None,
        help='folder to save the text or .json files',
    )
    parser.add_argument(
        '--save-ext',
        '-ext',
        type=str,
        choices=['.json', '.txt'],
        default='.txt',
        help='the format to save files in',
    )
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=10)
    parser.add_argument(
        '--save-by-subgroup',
        '-savbysubgroup',
        type='bool',
        default=True,
        help='Choose to save subgroups separately or not',
    )
    parser.set_defaults(batchsize=256)
    return parser


def print_report(world, log_time):
    total_cnt = world.num_examples()
    old_report = world.report()
    report = {key: old_report[key] for key in old_report if 'total_' in key}
    text, report = log_time.log(old_report.get('exs', 0), total_cnt, report)
    logging.info(text)


def _save_single_datatype(opt, agent, datatype):
    datatype_opt = opt.copy()
    datatype_opt['datatype'] = datatype
    world = create_task(datatype_opt, agent)
    save_by_subgroup = opt.get('save_by_subgroup', True)

    # set up logging
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    contents = defaultdict(list)
    DATA_IND, MODEL_IND = (0, 1)

    while not world.epoch_done():
        world.parley()
        acts = world.get_acts()
        if isinstance(acts[DATA_IND], dict):
            iter = [(acts[DATA_IND], acts[MODEL_IND])]
        else:
            iter = zip(acts[DATA_IND], acts[MODEL_IND])
        for data, model in iter:
            if data.get('batch_padding', False):
                break
            exs = {
                'text': data['text'],
                'labels': data['old_labels'][0],
                'subgroup_label': model['text'],
            }
            keys = exs['subgroup_label'] if save_by_subgroup else 'all'
            contents[keys].append(exs)

        if log_time.time() > log_every_n_secs or world.epoch_done():
            print_report(world, log_time)

    ext = opt['save_ext']
    for typename, subgroup_content in contents.items():
        typename = typename.split(':')[-1]
        filename = '_'.join([opt['save_loc'], datatype.split(':')[0], typename]) + ext
        with open(filename, 'w') as f:
            for dic in subgroup_content:
                line = ext_func[ext](dic)
                f.write(line + '\n')
        print('should be saved', filename)


def label_subgroup_saver(opt):
    if 'train' in opt['datatype'] and 'evalmode' not in opt['datatype']:
        raise ValueError(
            'You should use --datatype train:evalmode if you want to evaluate on '
            'the training set.'
        )
    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()

    if opt.get('save_loc') is None:
        opt['save_loc'] = opt['task'].replace('/', '_')

    datatypes = opt['datatype'].split(',')
    for datatype in datatypes:
        _save_single_datatype(opt, agent, datatype)


@register_script('label_subgroups_saver', aliases=['labelsav', 'lsav'])
class LabelSubgroupSaver(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return label_subgroup_saver(self.opt)
