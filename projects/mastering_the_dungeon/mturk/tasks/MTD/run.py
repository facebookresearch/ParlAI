#!/usr/bin/env python3

##
## Copyright (c) Facebook, Inc. and its affiliates.
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##

from parlai.core.params import ParlaiParser
import os
import time
from joblib import Parallel, delayed
from os.path import join
import pickle
import random
from copy import copy, deepcopy
from collections import defaultdict as dd
import traceback
import numpy as np
from projects.mastering_the_dungeon.projects.graph_world2.train import (
    additional_validate,
    ablation_exp,
)
import sys

import projects.mastering_the_dungeon as parlai_internal

sys.modules['parlai_internal'] = parlai_internal

START_TASK_TIMEOUT = 10 * 60
parent_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = join(parent_dir, 'checkpoint')
cur_dir = join(parent_dir, 'tmp')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(cur_dir):
    os.makedirs(cur_dir)


def print_and_log(s):
    print(s)
    f_log = open(join(cur_dir, 'mtd_log.txt'), 'a+')
    f_log.write(s + '\n')
    f_log.close()


def log_only(s):
    f_log = open(join(cur_dir, 'mtd_log.txt'), 'a+')
    f_log.write(s + '\n')
    f_log.close()


def get_output_dir(opt, round_index, version_num=None):
    output_dir = join(
        opt['datapath'], 'graph_world2_v{}_r{}'.format(version_num, round_index)
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def get_init_data(
    opt, base_only=False, delta_only=False, version_num=None, resplit=True
):
    RESPLIT_PREFIX = 'resplit_' if resplit else ''
    TRAIN_DIR, VALID_DIR = (
        join(opt['datapath'], 'graph_world2', 'train'),
        join(opt['datapath'], 'graph_world2', 'valid'),
    )

    if delta_only and opt['start_round'] > 1:
        output_dir = get_output_dir(
            opt, opt['start_round'] - 1, version_num=version_num
        )
        delta_train_data = pickle.load(
            open(
                join(output_dir, '{}delta_train_data.pkl'.format(RESPLIT_PREFIX)), 'rb'
            )
        )
        delta_valid_data = pickle.load(
            open(
                join(output_dir, '{}delta_valid_data.pkl'.format(RESPLIT_PREFIX)), 'rb'
            )
        )
        return delta_train_data, delta_valid_data

    def read_data(data_path):
        data = []
        for filename in os.listdir(data_path):
            if filename.endswith('pkl'):
                data.append(pickle.load(open(join(data_path, filename), 'rb')))
        return data

    train_data, valid_data = read_data(TRAIN_DIR), read_data(VALID_DIR)
    if base_only or delta_only:
        return train_data, valid_data

    for i in range(1, opt['start_round']):
        try:
            output_dir = get_output_dir(opt, i, version_num=version_num)
            delta_train_data = pickle.load(
                open(
                    join(output_dir, '{}delta_train_data.pkl'.format(RESPLIT_PREFIX)),
                    'rb',
                )
            )
            delta_valid_data = pickle.load(
                open(
                    join(output_dir, '{}delta_valid_data.pkl'.format(RESPLIT_PREFIX)),
                    'rb',
                )
            )
            train_data.extend(delta_train_data)
            valid_data.extend(delta_valid_data)
        except:
            print_and_log('Error: {}'.format(traceback.format_exc()))
    return train_data, valid_data


def get_rest_data(opt, version_num=None):
    output_dir = get_output_dir(opt, opt['start_round'] - 1, version_num=version_num)
    rest_data = pickle.load(open(join(output_dir, 'resplit_rest_data.pkl'), 'rb'))
    return rest_data


def overall_split(opt):
    VERSIONS = [13, 14, 15, 'BASELINE_2']

    min_for_round = [1000 for _ in range(5)]

    for version in VERSIONS:
        for round_index in range(1, 6):
            output_dir = get_output_dir(opt, round_index, version_num=version)
            filtered_data_list = pickle.load(
                open(join(output_dir, 'filtered_data_list.pkl'), 'rb')
            )
            min_for_round[round_index - 1] = min(
                min_for_round[round_index - 1], len(filtered_data_list)
            )

    def flatten(a):
        return [e for aa in a for e in aa]

    for round_index in range(1, 6):
        min_num = min_for_round[round_index - 1]
        train_num, test_num = int(min_num * 0.8), int(min_num * 0.2)
        for version in VERSIONS:
            output_dir = get_output_dir(opt, round_index, version_num=version)
            filtered_data_list = pickle.load(
                open(join(output_dir, 'filtered_data_list.pkl'), 'rb')
            )
            random.seed(13)
            random.shuffle(filtered_data_list)
            train_list, test_list, rest_list = (
                flatten(filtered_data_list[:train_num]),
                flatten(filtered_data_list[train_num : train_num + test_num]),
                flatten(filtered_data_list[train_num + test_num :]),
            )
            pickle.dump(
                train_list, open(join(output_dir, 'resplit_delta_train_data.pkl'), 'wb')
            )
            pickle.dump(
                test_list, open(join(output_dir, 'resplit_delta_valid_data.pkl'), 'wb')
            )
            pickle.dump(
                rest_list, open(join(output_dir, 'resplit_rest_data.pkl'), 'wb')
            )


def train(
    opt,
    round_index,
    machine_index,
    file_index,
    train_data,
    valid_data,
    valid_weights=None,
    save_all=True,
    return_acc_len=False,
    seq2seq=False,
):
    if valid_weights is not None:
        assert len(valid_data) == len(valid_weights), (
            len(valid_data),
            len(valid_weights),
        )

    train_filename = join(cur_dir, '{}_{}_train.pkl'.format(round_index, file_index))
    valid_filename = join(cur_dir, '{}_{}_valid.pkl'.format(round_index, file_index))
    out_filename = join(cur_dir, '{}_{}_out.txt'.format(round_index, file_index))

    pickle.dump(train_data, open(train_filename, 'wb'))
    pickle.dump(valid_data, open(valid_filename, 'wb'))

    if valid_weights is not None:
        weight_filename = join(
            cur_dir, '{}_{}_weights.pkl'.format(round_index, file_index)
        )
        pickle.dump(valid_weights, open(weight_filename, 'wb'))
    else:
        weight_filename = ''

    if save_all:
        model_filename = join(
            cur_dir, '{}_{}_model.pkl'.format(round_index, file_index)
        )
        data_agent_filename = join(
            cur_dir, '{}_{}_data_agent.pkl'.format(round_index, file_index)
        )
        wrong_data_filename = join(
            cur_dir, '{}_{}_wrong_data.pkl'.format(round_index, file_index)
        )
    else:
        model_filename = ''
        data_agent_filename = ''
        wrong_data_filename = ''

    new_opt = {
        'max_iter': opt['max_iter'],
        'num_runs': opt['num_runs'],
        'train_data_file': train_filename,
        'valid_data_file': valid_filename,
        'perf_out_file': out_filename,
        'weight_file': weight_filename,
        'model_file': model_filename,
        'data_agent_file': data_agent_filename,
        'wrong_data_file': wrong_data_filename,
        'task': 'projects.mastering_the_dungeon.tasks.graph_world2.agents',
        'batchsize': 1,
        'rnn_h': opt['rnn_h'],
        'embedding_dim': opt['embedding_dim'],
        'cuda': True,
        'seq2seq': seq2seq,
    }
    os.chdir(parent_dir)

    job_num = machine_index % opt['num_machines']
    job_in_file = '../../../projects/graph_world2/job-in-{}.pkl'.format(job_num)
    job_out_file = '../../../projects/graph_world2/job-out-{}.txt'.format(job_num)

    if os.path.isfile(job_out_file):
        os.remove(job_out_file)
    if os.path.isfile(job_in_file):
        os.remove(job_in_file)

    with open(job_in_file, 'wb') as f_job_in:
        pickle.dump(new_opt, f_job_in)

    start_time = time.time()
    while True:
        time.sleep(5)
        if not os.path.isfile(job_in_file):
            break
        if time.time() - start_time > 60 * 5:
            try:
                os.remove(job_in_file)
            except:
                pass
            print_and_log('job {} timeout'.format(job_num))
            if return_acc_len:
                return 0.0, [0.0 for _ in range(4)]
            return 0.0

    start_time = time.time()
    while True:
        time.sleep(5)
        if time.time() - start_time > opt['job_timeout']:
            print_and_log(
                'job {} tiemout after {} seconds, bad gpu'.format(
                    job_num, opt['job_timeout']
                )
            )
            if return_acc_len:
                return 0.0, [0.0 for _ in range(4)]
            return 0.0

        if os.path.isfile(job_out_file):
            with open(job_out_file) as f_job_out:
                result = f_job_out.read().strip()
            os.remove(job_out_file)
            print_and_log('job {} exited: {}'.format(job_num, result))
            if result != '0':
                print_and_log('job {} abandoned'.format(job_num))
                if return_acc_len:
                    return 0.0, [0.0 for _ in range(4)]
                return 0.0
            try:
                f_perf = open(out_filename)
                perf = float(f_perf.readline().strip())
                f_perf.close()
                print_and_log('job {} perf: {}'.format(job_num, perf))
                if return_acc_len:
                    f_acc_len = open(out_filename + '.len')
                    acc_len = list(map(float, f_acc_len.readline().strip().split()))
                    f_acc_len.close()
                    print_and_log('job {} acc_len: {}'.format(job_num, acc_len))
                    return perf, acc_len
                return perf
            except:
                print_and_log('Error: {}'.format(traceback.format_exc()))
                if return_acc_len:
                    return 0.0, [0.0 for _ in range(4)]
                return 0.0


def batch_train(
    opt,
    round_index,
    round_train_data,
    round_valid_data,
    round_valid_weights=None,
    save_all=True,
    file_indices=None,
    return_acc_len=False,
    seq2seq=False,
):
    i = 0
    perfs = []
    M = len(round_train_data)
    while i < M:
        j = min(i + opt['num_machines'], M)
        cur_perfs = Parallel(n_jobs=j - i, backend='threading')(
            delayed(train)(
                opt,
                round_index,
                train_index,
                file_indices[train_index] if file_indices else train_index,
                round_train_data[train_index],
                round_valid_data[train_index],
                valid_weights=round_valid_weights[train_index]
                if round_valid_weights
                else None,
                save_all=save_all,
                return_acc_len=return_acc_len,
                seq2seq=seq2seq,
            )
            for train_index in range(i, j)
        )
        perfs.extend(cur_perfs)
        i = j

    error_indices, valid_indices = [], []
    for i, perf in enumerate(perfs):
        if perf == 0.0 or type(perf) == tuple and perf[0] == 0.0:
            error_indices.append(i)
        elif i < opt['num_machines']:
            valid_indices.append(i)

    M = len(error_indices)
    TMP_NUM_MACHINES = len(valid_indices)
    if M > 0 and TMP_NUM_MACHINES > 0:
        i = 0
        error_perfs = []
        while i < M:
            j = min(i + TMP_NUM_MACHINES, M)
            cur_perfs = Parallel(n_jobs=j - i, backend='threading')(
                delayed(train)(
                    opt,
                    round_index,
                    valid_indices[train_index],
                    file_indices[error_indices[train_index]]
                    if file_indices
                    else error_indices[train_index],
                    round_train_data[error_indices[train_index]],
                    round_valid_data[error_indices[train_index]],
                    valid_weights=round_valid_weights[error_indices[train_index]]
                    if round_valid_weights
                    else None,
                    save_all=save_all,
                    return_acc_len=return_acc_len,
                    seq2seq=seq2seq,
                )
                for train_index in range(i, j)
            )
            error_perfs.extend(cur_perfs)
            i = j
        for i in range(M):
            perfs[error_indices[i]] = error_perfs[i]

    return perfs


def batch_valid(opt, round_index, constrain_=True):
    perfs = []
    for i in range(100000):
        model_filename = join(cur_dir, '{}_{}_model.pkl'.format(round_index, i))
        if not os.path.exists(model_filename):
            break
        data_agent_filename = join(
            cur_dir, '{}_{}_data_agent.pkl'.format(round_index, i)
        )
        valid_filename = join(cur_dir, '{}_{}_valid.pkl'.format(round_index, i))
        model = pickle.load(open(model_filename, 'rb'))
        data_agent = pickle.load(open(data_agent_filename, 'rb'))
        perf = additional_validate(
            opt, model, data_agent, valid_filename, constrain_=constrain_
        )
        print('batch_valid {} {}'.format(round_index, i))
        perfs.append(perf)
    return perfs


def batch_valid_with_data(
    opt, round_index, file_index, valid_filename, constrain_=True
):
    model_filename = join(cur_dir, '{}_{}_model.pkl'.format(round_index, file_index))
    data_agent_filename = join(
        cur_dir, '{}_{}_data_agent.pkl'.format(round_index, file_index)
    )
    model = pickle.load(open(model_filename, 'rb'))
    data_agent = pickle.load(open(data_agent_filename, 'rb'))
    perf = additional_validate(
        opt, model, data_agent, valid_filename, constrain_=constrain_, no_hits=True
    )
    return perf


def overall_ablation(opt):
    VERSIONS = [13, 14, 15, 'BASELINE_2']
    NAMES = ['MTD LIMIT', 'MTD', 'MTD NO MODEL FEEDBACK', 'BASELINE']

    all_train_data, all_valid_data, all_rest_data = [], [], []
    for v_id, version in enumerate(VERSIONS):
        version_train_data, version_valid_data, version_rest_data = [], [], []
        for i in range(2, 7):
            final_opt = deepcopy(opt)
            final_opt['start_round'] = i
            final_opt['datapath'] = final_opt['datapath'].replace('/data', '/new_data')
            cur_train_data, cur_valid_data = get_init_data(
                final_opt, delta_only=True, version_num=version, resplit=True
            )
            cur_rest_data = get_rest_data(final_opt, version_num=version)
            version_train_data.extend(cur_train_data)
            version_valid_data.extend(cur_valid_data)
            version_rest_data.extend(cur_rest_data)
        print(
            '{}: train {} test {} rest {}'.format(
                NAMES[v_id],
                len(version_train_data),
                len(version_valid_data),
                len(version_rest_data),
            )
        )
        all_train_data.append(version_train_data)
        all_valid_data.append(version_valid_data)
        all_rest_data.append(version_rest_data)

    init_train_data, init_valid_data = get_init_data(opt, base_only=True)
    init_valid_data.extend(init_train_data)
    print('init: test {}'.format(len(init_valid_data)))

    final_valid_data = copy(init_valid_data)
    for i in range(len(all_rest_data)):
        final_valid_data.extend(all_valid_data[i])
        final_valid_data.extend(all_rest_data[i])
    print('all: test {}'.format(len(final_valid_data)))

    final_train_data = all_train_data[1]

    train_filename = join(cur_dir, 'ABLATION_train.pkl')
    valid_filename = join(cur_dir, 'ABLATION_valid.pkl')

    pickle.dump(final_train_data, open(train_filename, 'wb'))
    pickle.dump(final_valid_data, open(valid_filename, 'wb'))

    model_filename = join(cur_dir, 'ABLATION_model.pkl')
    data_agent_filename = join(cur_dir, 'ABLATION_data_agent.pkl')
    wrong_data_filename = join(cur_dir, 'ABLATION_wrong_data.pkl')

    names = ['ac', 'ac - counter', 'ac - counter - room', 'seq2seq']
    seq2seq_options = [False, False, False, True]
    counter_ablations = [False, True, True, False]
    room_ablations = [False, False, True, False]

    for i in range(1, 3):
        new_opt = deepcopy(opt)
        new_opt.update(
            {
                'max_iter': opt['max_iter'],
                'num_runs': 1,
                'train_data_file': train_filename,
                'valid_data_file': valid_filename,
                'model_file': model_filename,
                'data_agent_file': data_agent_filename,
                'wrong_data_file': wrong_data_filename,
                'task': 'parlai_internal.tasks.graph_world2.agents',
                'batchsize': 1,
                'rnn_h': opt['rnn_h'],
                'embedding_dim': opt['embedding_dim'],
                'cuda': True,
                'seq2seq': seq2seq_options[i],
                'counter_ablation': counter_ablations[i],
                'room_ablation': room_ablations[i],
                'weight_file': '',
                'datatype': 'train',
            }
        )
        perf = ablation_exp(new_opt)
        print(names[i], perf)


def overall_run(opt, seq2seq=False):
    VERSIONS = [13, 14, 15, 'BASELINE_2']
    NAMES = ['MTD LIMIT', 'MTD', 'MTD NO MODEL FEEDBACK', 'BASELINE']

    all_train_data, all_valid_data, all_rest_data = [], [], []
    for v_id, version in enumerate(VERSIONS):
        version_train_data, version_valid_data, version_rest_data = [], [], []
        # for i in range(2, 7):
        for i in range(2, 3):
            final_opt = deepcopy(opt)
            final_opt['start_round'] = i
            cur_train_data, cur_valid_data = get_init_data(
                final_opt, delta_only=True, version_num=version, resplit=True
            )
            cur_rest_data = get_rest_data(final_opt, version_num=version)
            version_train_data.extend(cur_train_data)
            version_valid_data.extend(cur_valid_data)
            version_rest_data.extend(cur_rest_data)
        print(
            '{}: train {} test {} rest {}'.format(
                NAMES[v_id],
                len(version_train_data),
                len(version_valid_data),
                len(version_rest_data),
            )
        )
        all_train_data.append(version_train_data)
        all_valid_data.append(version_valid_data)
        all_rest_data.append(version_rest_data)

    init_train_data, init_valid_data = get_init_data(opt, base_only=True)
    init_valid_data.extend(init_train_data)
    print('init: test {}'.format(len(init_valid_data)))

    final_valid_data = copy(init_valid_data)
    for i in range(len(all_rest_data)):
        final_valid_data.extend(all_valid_data[i])
        final_valid_data.extend(all_rest_data[i])
    print('all: test {}'.format(len(final_valid_data)))

    round_train_data, round_valid_data = [], []
    M = opt['num_runs']
    for cur_train_data in all_train_data:
        for cur_valid_data in [init_valid_data, final_valid_data]:
            for _ in range(M):
                round_train_data.append(cur_train_data)
                round_valid_data.append(cur_valid_data)

    final_opt = deepcopy(opt)
    final_opt['num_runs'] = 1
    perfs = batch_train(
        final_opt,
        'OVERALL_TEST' if not seq2seq else 'SEQ2SEQ_TEST',
        round_train_data,
        round_valid_data,
        save_all=True,
        return_acc_len=True,
        seq2seq=seq2seq,
    )

    def get_acc_and_acc_len(perfs):
        acc, acc_len, cnts = 0.0, [0.0 for _ in range(4)], 0
        accs = []
        for i, (cur_acc, cur_acc_len) in enumerate(perfs):
            if cur_acc == 0:
                continue
            acc += cur_acc
            accs.append(cur_acc)
            for j in range(4):
                acc_len[j] += cur_acc_len[j]
            cnts += 1
        if cnts == 0:
            return acc, acc_len
        acc /= cnts
        for j in range(4):
            acc_len[j] /= cnts
        stddev = np.std(np.array(accs), dtype=np.float64)
        return acc, acc_len, stddev

    start, end = 0, M
    for train_name in NAMES:
        for valid_name in ['INIT', 'ALL']:
            sub_perfs = perfs[start:end]
            acc, acc_len, stddev = get_acc_and_acc_len(sub_perfs)
            print_and_log(
                '{} on {}: acc {} stddev {} acc_len {}'.format(
                    train_name, valid_name, acc, stddev, acc_len
                )
            )
            log_only('{} on {}: {}'.format(train_name, valid_name, sub_perfs))
            start = end
            end = start + M


def overall_run_data_breakdown(opt, seq2seq=False):
    VERSIONS = [13, 14, 15, 'BASELINE_2']
    NAMES = ['MTD LIMIT', 'MTD', 'MTD NO MODEL FEEDBACK', 'BASELINE']

    all_train_data, all_valid_data, all_rest_data = [], [], []
    for v_id, version in enumerate(VERSIONS):
        version_train_data, version_valid_data, version_rest_data = [], [], []
        for i in range(2, 7):
            final_opt = deepcopy(opt)
            final_opt['start_round'] = i
            cur_train_data, cur_valid_data = get_init_data(
                final_opt, delta_only=True, version_num=version, resplit=True
            )
            cur_rest_data = get_rest_data(final_opt, version_num=version)
            version_train_data.extend(cur_train_data)
            version_valid_data.extend(cur_valid_data)
            version_rest_data.extend(cur_rest_data)
        print(
            '{}: train {} test {} rest {}'.format(
                NAMES[v_id],
                len(version_train_data),
                len(version_valid_data),
                len(version_rest_data),
            )
        )
        all_train_data.append(version_train_data)
        all_valid_data.append(version_valid_data)
        all_rest_data.append(version_rest_data)

    init_train_data, init_valid_data = get_init_data(opt, base_only=True)
    init_valid_data.extend(init_train_data)
    print('init: test {}'.format(len(init_valid_data)))

    final_valid_data = copy(init_valid_data)
    for i in range(len(all_rest_data)):
        final_valid_data.extend(all_valid_data[i])
        final_valid_data.extend(all_rest_data[i])
    print('all: test {}'.format(len(final_valid_data)))

    round_train_data, round_valid_data = [], []
    M = opt['num_runs']
    for cur_train_data in all_train_data:
        for cur_valid_data in [init_valid_data, final_valid_data]:
            for _ in range(M):
                round_train_data.append(cur_train_data)
                round_valid_data.append(cur_valid_data)

    def get_scores_and_std(perfs):
        def _get_mean(l):
            return sum(l) * 1.0 / len(l)

        def _get_std(l):
            return np.std(np.array(l), dtype=np.float64)

        ret = dd(list)
        for perf in perfs:
            for k, v in perf.items():
                ret[k].append(v)
        final_ret = {}
        for k, v in ret.items():
            final_ret[k + '_mean'] = _get_mean(v)
            final_ret[k + '_std'] = _get_std(v)
        return final_ret

    file_index = 0
    my_valid_data = [
        all_valid_data[0] + all_rest_data[0],
        all_valid_data[-1] + all_rest_data[-1],
    ]
    my_valid_names = ['MTD', 'BASELINE']
    for train_name in NAMES:
        for valid_name in ['INIT', 'ALL']:
            if train_name not in ['MTD', 'BASELINE'] or valid_name == 'INIT':
                file_index += M
                continue
            for valid_id in range(2):
                valid_filename = join(cur_dir, 'DATA_BREAK_data.pkl')
                pickle.dump(my_valid_data[valid_id], open(valid_filename, 'wb'))
                perfs = []
                for m in range(M):
                    perf = batch_valid_with_data(
                        opt,
                        'OVERALL_TEST' if not seq2seq else 'SEQ2SEQ_TEST',
                        file_index + m,
                        valid_filename,
                        constrain_=True,
                    )
                    perfs.append(perf)
                scores = get_scores_and_std(perfs)
                print_and_log(
                    '{} on {}: {}'.format(train_name, my_valid_names[valid_id], scores)
                )
            file_index += M


def overall_add_val(opt, seq2seq=False, constrain_=True):
    NAMES = ['MTD LIMIT', 'MTD', 'MTD NO MODEL FEEDBACK', 'BASELINE']

    def get_scores_and_std(perfs):
        def _get_mean(l):
            return sum(l) * 1.0 / len(l)

        def _get_std(l):
            return np.std(np.array(l), dtype=np.float64)

        ret = dd(list)
        for perf in perfs:
            for k, v in perf.items():
                ret[k].append(v)
        final_ret = {}
        for k, v in ret.items():
            final_ret[k + '_mean'] = _get_mean(v)
            final_ret[k + '_std'] = _get_std(v)
        return final_ret

    perfs = batch_valid(
        opt, 'OVERALL_TEST' if not seq2seq else 'SEQ2SEQ_TEST', constrain_=constrain_
    )
    M = opt['num_runs']
    start, end = 0, M
    for train_name in NAMES:
        for valid_name in ['INIT', 'ALL']:
            sub_perfs = perfs[start:end]
            scores = get_scores_and_std(sub_perfs)
            print_and_log('{} on {}: {}'.format(train_name, valid_name, scores))
            start = end
            end = start + M


def overall_run_rounds_breakdown(opt, seq2seq=False):
    VERSIONS = [13, 14, 15, 'BASELINE_2']
    NAMES = ['MTD LIMIT', 'MTD', 'MTD NO MODEL FEEDBACK', 'BASELINE']

    all_train_data, all_valid_data, all_rest_data = [], [], []
    for v_id, version in enumerate(VERSIONS):
        version_train_data, version_valid_data, version_rest_data = [], [], []
        for i in range(2, 7):
            final_opt = deepcopy(opt)
            final_opt['start_round'] = i
            cur_train_data, cur_valid_data = get_init_data(
                final_opt, delta_only=True, version_num=version, resplit=True
            )
            cur_rest_data = get_rest_data(final_opt, version_num=version)
            version_train_data.append(cur_train_data)
            version_valid_data.extend(cur_valid_data)
            version_rest_data.extend(cur_rest_data)
        print(
            '{}: train {} test {} rest {}'.format(
                NAMES[v_id],
                len(version_train_data),
                len(version_valid_data),
                len(version_rest_data),
            )
        )
        all_train_data.append(version_train_data)
        all_valid_data.append(version_valid_data)
        all_rest_data.append(version_rest_data)

    init_train_data, init_valid_data = get_init_data(opt, base_only=True)
    init_valid_data.extend(init_train_data)
    print('init: test {}'.format(len(init_valid_data)))

    final_valid_data = copy(init_valid_data)
    for i in range(len(all_rest_data)):
        final_valid_data.extend(all_valid_data[i])
        final_valid_data.extend(all_rest_data[i])
    print('all: test {}'.format(len(final_valid_data)))

    round_train_data, round_valid_data = [], []
    M = opt['num_runs']
    for cur_train_data in all_train_data:
        for cur_valid_data in [init_valid_data, final_valid_data]:
            for round_index in range(5):
                tmp_train_data = []
                for tmp_round_index in range(round_index + 1):
                    tmp_train_data.extend(cur_train_data[tmp_round_index])
                for _ in range(M):
                    round_train_data.append(tmp_train_data)
                    round_valid_data.append(cur_valid_data)

    final_opt = deepcopy(opt)
    final_opt['num_runs'] = 1
    perfs = batch_train(
        final_opt,
        'OVERALL_ROUND_BREAK' if not seq2seq else 'SEQ2SEQ_ROUND_BREAK',
        round_train_data,
        round_valid_data,
        save_all=True,
        return_acc_len=True,
        seq2seq=seq2seq,
    )

    def get_acc_and_acc_len(perfs):
        acc, acc_len, cnts = 0.0, [0.0 for _ in range(4)], 0
        accs = []
        for i, (cur_acc, cur_acc_len) in enumerate(perfs):
            if cur_acc == 0:
                continue
            acc += cur_acc
            accs.append(cur_acc)
            for j in range(4):
                acc_len[j] += cur_acc_len[j]
            cnts += 1
        if cnts == 0:
            return acc, acc_len
        acc /= cnts
        for j in range(4):
            acc_len[j] /= cnts
        stddev = np.std(np.array(accs), dtype=np.float64)
        return acc, acc_len, stddev

    start, end = 0, M
    for train_name in NAMES:
        for valid_name in ['INIT', 'ALL']:
            for round_index in range(5):
                sub_perfs = perfs[start:end]
                acc, acc_len, stddev = get_acc_and_acc_len(sub_perfs)
                print_and_log(
                    '{} on {} round{}: acc {} stddev {} acc_len {}'.format(
                        train_name, valid_name, round_index, acc, stddev, acc_len
                    )
                )
                log_only(
                    '{} on {} round {}: {}'.format(
                        train_name, valid_name, round_index, sub_perfs
                    )
                )
                start = end
                end = start + M


if __name__ == '__main__':
    argparser = ParlaiParser(False, False)

    # ============ below copied from projects/graph_world2/train.py ============
    argparser.add_arg('--vocab_size', type=int, default=1000)
    argparser.add_arg('--terminate', type=bool, default=False)
    argparser.add_arg('--lr', type=float, default=1e-3)
    argparser.add_arg('--max_seq_in', type=int, default=30)
    argparser.add_arg('--embedding_dim', type=int, default=50)
    argparser.add_arg('--rnn_h', type=int, default=350)
    argparser.add_arg('--rnn_layers', type=int, default=1)
    argparser.add_arg('--cuda', type=bool, default=True)
    argparser.add_arg('--eval_period', type=int, default=200)
    argparser.add_arg('--max_seq_out', type=int, default=5)
    argparser.add_arg('--label_ratio', type=float, default=1.0)
    argparser.add_arg('--max_iter', type=int, default=100000)
    argparser.add_arg('--exit_iter', type=int, default=3000)
    argparser.add_arg('--num_runs', type=int, default=10)

    argparser.add_arg('--train_data_file', type=str, default='')
    argparser.add_arg('--valid_data_file', type=str, default='')
    argparser.add_arg('--perf_out_file', type=str, default='')
    argparser.add_arg('--weight_file', type=str, default='')
    argparser.add_arg('--model_file', type=str, default='')
    argparser.add_arg('--data_agent_file', type=str, default='')
    argparser.add_arg('--wrong_data_file', type=str, default='')

    argparser.add_arg('--once', type=bool, default=False)
    argparser.add_arg('--job_num', type=int)

    argparser.add_arg('--counter_ablation', type=bool, default=False)
    argparser.add_arg('--room_ablation', type=bool, default=False)
    # ============ above copied from projects/graph_world2/train.py ============

    argparser.add_argument('--num_machines', type=int, default=1)
    argparser.add_argument('--job_timeout', type=float, default=3600 * 4)

    argparser.add_argument('--split', action='store_true', default=False)
    argparser.add_argument('--train', action='store_true', default=False)
    argparser.add_argument('--eval', action='store_true', default=False)
    argparser.add_argument('--seq2seq', action='store_true', default=False)
    argparser.add_argument('--constrain', action='store_true', default=False)
    argparser.add_argument('--rounds_breakdown', action='store_true', default=False)
    argparser.add_argument('--data_breakdown', action='store_true', default=False)
    argparser.add_argument('--ablation', action='store_true', default=False)

    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()

    # ============ below copied from projects/graph_world2/train.py ============
    opt['bidir'] = True
    opt['action_type_emb_dim'] = 5
    opt['counter_max'] = 3
    opt['counter_emb_dim'] = 5
    # ============ above copied from projects/graph_world2/train.py ============

    if opt['split']:
        overall_split(opt)
        quit()
    if opt['train']:
        overall_run(opt, seq2seq=opt['seq2seq'])
        quit()
    if opt['eval']:
        overall_add_val(opt, seq2seq=opt['seq2seq'], constrain_=opt['constrain'])
        quit()
    if opt['rounds_breakdown']:
        overall_run_rounds_breakdown(opt, seq2seq=opt['seq2seq'])
        quit()
    if opt['data_breakdown']:
        overall_run_data_breakdown(opt, seq2seq=opt['seq2seq'])
        quit()
    if opt['ablation']:
        overall_ablation(opt)
        quit()
