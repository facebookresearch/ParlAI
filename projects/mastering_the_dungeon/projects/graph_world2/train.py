#!/usr/bin/env python3

##
## Copyright (c) Facebook, Inc. and its affiliates.
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##

from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from projects.mastering_the_dungeon.agents.graph_world2.agents import ObjectChecklistDataAgent, ObjectChecklistModelAgent, Seq2SeqDataAgent, Seq2SeqModelAgent
from projects.mastering_the_dungeon.agents.graph_world2.models import Seq2SeqModel
from copy import deepcopy
import os
import sys
import torch
from projects.mastering_the_dungeon.tasks.graph_world2.graph import Graph
import statistics
import pickle
import time
import traceback
import scipy.stats as ss
from torch.autograd import Variable
import random

import projects.mastering_the_dungeon as parlai_internal
sys.modules['parlai_internal'] = parlai_internal

def prepro(opt):
    agent = ObjectChecklistDataAgent(opt) if not opt['seq2seq'] else Seq2SeqDataAgent(opt)

    opt = deepcopy(opt)
    opt['datatype'] = 'train'
    opt['terminate'] = True
    opt['batchsize'] = 1
    world = create_task(opt, agent)

    for _ in world:
        world.parley()

    agent.build()
    return agent

def validate(opt, agent):
    old_datatype = agent.opt['datatype']
    agent.opt['datatype'] = 'valid'

    opt = deepcopy(opt)
    opt['datatype'] = 'valid'
    opt['terminate'] = True
    opt['batchsize'] = 1

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    valid_world = create_task(opt, agent)
    sys.stdout = old_stdout

    for _ in valid_world:
        valid_world.parley()

    stats = valid_world.report()
    agent.opt['datatype'] = old_datatype
    return stats

def get_metrics(actions, gt_actions):
    tp, fp, fn = 0, 0, 0
    action_set, gt_action_set = set(actions), set(gt_actions)
    for action in action_set:
        if action in gt_action_set:
            tp += 1
        else:
            fp += 1
    for action in gt_action_set:
        if action not in action_set:
            fn += 1
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    return f1

def get_accuracy(actions, gt_actions, graph):
    graph_a, graph_b = graph.copy(), graph.copy()
    graph_a.parse_exec(' '.join(actions))
    graph_b.parse_exec(' '.join(gt_actions))
    return float(graph_a == graph_b)

def additional_validate(opt, model, data_agent, valid_data_file, constrain_=True, no_hits=False):
    seq2seq = isinstance(model, Seq2SeqModel)

    def _get_actions(inst, symb_points):
        ret = []
        for i in range(len(symb_points) - 1):
            ret.append(' '.join(inst[symb_points[i]: symb_points[i + 1]]))
        return ret

    def _get_variable(np_a, volatile=False):
        if opt['cuda']:
            return Variable(torch.from_numpy(np_a), volatile=volatile).cuda()
        return Variable(torch.from_numpy(np_a), volatile=volatile)

    if not seq2seq:
        check_mapping = _get_variable(data_agent.get_check_mapping(), True)
    valid_data = pickle.load(open(valid_data_file, 'rb'))

    all_f1 = 0
    all_gt_actions = []
    all_accuracy = 0

    for example in valid_data:
        exp_dict = {'text': example[2], 'actions': example[3], 'graph': example[1]}
        if not seq2seq:
            x, action_key, second_action_key, action_type, current_room, checked, y, y_mask, counter_feat = data_agent.get_data([exp_dict], 'valid')
            x, action_key, second_action_key, action_type, checked = _get_variable(x, True), _get_variable(action_key, True), _get_variable(second_action_key, True), _get_variable(action_type, True), _get_variable(checked, True)
            text_out = model.forward_predict(x, action_key, second_action_key, action_type, check_mapping, checked, [example[1]], data_agent, constrain_=constrain_)[0]
        else:
            x, y = data_agent.get_data([exp_dict], 'valid')
            x, y = _get_variable(x, True), _get_variable(y, True)
            text_out = model.forward_predict(x, [example[1]], data_agent, constrain_=constrain_)[0]
        actions = text_out[: -1]
        gt_actions = _get_actions(*Graph.parse_static(example[3]))
        cur_f1 = get_metrics(actions, gt_actions)
        all_f1 += cur_f1
        all_accuracy += get_accuracy(actions, gt_actions, example[1])

        all_gt_actions.append(gt_actions)

    if constrain_ and not no_hits:
        random.seed(13)
        hits1, hits5, hits10 = 0, 0, 0
        for i, example in enumerate(valid_data):
            all_dicts = []
            for j in range(100):
                idx = i if j == 99 else random.randint(0, len(valid_data) - 1)
                exp_dict = {'text': example[2], 'actions': ' '.join(all_gt_actions[idx]), 'graph': example[1]}
                all_dicts.append(exp_dict)
            if not seq2seq:
                x, action_key, second_action_key, action_type, current_room, checked, y, y_mask, counter_feat = data_agent.get_data(all_dicts, 'train', assert_=False)
                x, action_key, second_action_key, action_type, current_room, checked, y, y_mask, counter_feat = _get_variable(x, True), _get_variable(action_key, True), _get_variable(second_action_key, True), _get_variable(action_type, True), _get_variable(current_room, True), _get_variable(checked, True), _get_variable(y, True), _get_variable(y_mask, True), _get_variable(counter_feat, True)
                all_losses = model.forward_loss(x, action_key, second_action_key, action_type, current_room, checked, y, y_mask, counter_feat, average_=False).data.cpu().numpy()
            else:
                x, y = data_agent.get_data(all_dicts, 'train', assert_=False)
                x, y = _get_variable(x, True), _get_variable(y, True)
                all_losses = model.forward_loss(x, y, average_=False).data.cpu().numpy()
            ranks = ss.rankdata(all_losses, method='ordinal')
            if ranks[-1] == 1:
                hits1 += 1
            if ranks[-1] <= 5:
                hits5 += 1
            if ranks[-1] <= 10:
                hits10 += 1

    N = len(valid_data)
    if constrain_ and not no_hits:
        return {'accuracy': all_accuracy / N, 'f1': all_f1 / N, 'hits1': hits1 / N, 'hits5': hits5 / N, 'hits10': hits10 / N}
    return {'f1': all_f1 / N, 'accuracy': all_accuracy / N}

def log_print(s, out_file):
    print(s)
    if out_file is None:
        return
    f_log = open(out_file, 'a+')
    f_log.write(s + '\n')
    f_log.close()

def main(opt, return_full=False, out_file=None):
    data_agent = prepro(opt)
    model_agent = ObjectChecklistModelAgent(opt, data_agent=data_agent) if not opt['seq2seq'] else Seq2SeqModelAgent(opt, data_agent=data_agent)

    train_world = create_task(opt, model_agent)

    max_dict = model_agent.model.state_dict()

    max_acc, max_f1, max_data, last_max, max_acc_len = -1, 0, None, 0, None
    for iter in range(opt['max_iter']):
        if iter - last_max > opt['exit_iter']: break

        if 'inc_ratio' in opt and opt['inc_ratio'] > 0 and iter == opt['inc_pre_iters']:
            print('resetting best model for finetuning')
            model_agent.model.load_state_dict(max_dict)
            max_acc = 0

        train_world.parley()
        train_report = train_world.report()

        if iter % opt['eval_period'] == 0:
            stats = validate(opt, model_agent)
            cur_acc, cur_f1 = stats['acc'] / stats['cnt'], stats['f1'] / stats['cnt']
            if cur_acc > max_acc:
                max_acc = cur_acc
                max_data = (stats['correct_data'], stats['wrong_data'])
                max_dict = deepcopy(model_agent.model)
                last_max = iter
                max_acc_len = []
                for i in range(1, 5):
                    max_acc_len.append(stats['acc_len'][i] / stats['cnt_len'][i])
            max_f1 = max(max_f1, cur_f1)
            s = '#{} train {:.4f} valid {:.4f} acc {:.4f} f1 {:.4f} max_acc {:.4f} max_f1 {:.4f} acc_len'.format(iter, train_report['loss'], stats['loss'] / stats['cnt'], cur_acc, cur_f1, max_acc, max_f1)
            for i in range(1, 5):
                s += ' {:.4f}'.format(stats['acc_len'][i] / stats['cnt_len'][i])
            s += ' cnt_len'
            for i in range(1, 5):
                s += ' {:d}'.format(int(stats['cnt_len'][i]))
            log_print(s, out_file)

    wrong_data = max_data[1]

    if not return_full:
        return max_acc
    else:
        return max_acc, max_dict, model_agent.data_agent, wrong_data, max_acc_len

def online_exp(opt, log_file=None):
    model_dict, data_agent, wrong_data, max_acc_len = None, None, None, None
    max_accs = []
    record = -1.0
    for _ in range(opt['num_runs']):
        max_acc, cur_model_dict, cur_data_agent, cur_wrong_data, cur_max_acc_len = main(opt, True, log_file)
        max_accs.append(max_acc)
        if max_acc > record:
            record = max_acc
            model_dict, data_agent, wrong_data = cur_model_dict, cur_data_agent, cur_wrong_data
        if max_acc_len is None:
            max_acc_len = cur_max_acc_len
        else:
            for i in range(len(max_acc_len)):
                max_acc_len[i] += cur_max_acc_len[i]
    for i in range(len(max_acc_len)):
        max_acc_len[i] /= opt['num_runs']

    if opt['perf_out_file'] != '':
        fout = open(opt['perf_out_file'], 'w')
        fout.write('{}\n'.format(statistics.mean(max_accs)))
        fout.close()
        fout = open(opt['perf_out_file'] + '.len', 'w')
        fout.write('{}\n'.format(' '.join(list(map(str, max_acc_len)))))
        fout.close()
    if opt['model_file'] != '':
        pickle.dump(model_dict, open(opt['model_file'], 'wb'))
    if opt['data_agent_file'] != '':
        pickle.dump(data_agent, open(opt['data_agent_file'], 'wb'))
    if opt['wrong_data_file'] != '':
        pickle.dump(wrong_data, open(opt['wrong_data_file'], 'wb'))

def ablation_exp(opt):
    max_acc, cur_model_dict, cur_data_agent, cur_wrong_data, cur_max_acc_len = main(opt, True, None)
    return additional_validate(opt, cur_model_dict, cur_data_agent, opt['valid_data_file'], constrain_=True, no_hits=True)

if __name__ == '__main__':
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    argparser = ParlaiParser()
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
    argparser.add_arg('--seq2seq', type=bool, default=False)
    argparser.add_arg('--job_num', type=int)

    argparser.add_arg('--counter_ablation', type=bool, default=False)
    argparser.add_arg('--room_ablation', type=bool, default=False)

    opt = argparser.parse_args()
    
    opt['bidir'] = True
    opt['action_type_emb_dim'] = 5
    opt['counter_max'] = 3
    opt['counter_emb_dim'] = 5

    if opt['once']:
        online_exp(opt)
        quit()

    job_num = opt['job_num']
    input_file = 'job-in-{}.pkl'.format(job_num)
    output_file = 'job-out-{}.txt'.format(job_num)
    log_file = 'job-log-{}.txt'.format(job_num)

    while True:
        time.sleep(5)
        if os.path.isfile(input_file):
            time.sleep(5)
            log_print('grab job {}'.format(job_num),log_file)
            try:
                with open(input_file, 'rb') as f_in:
                    job_in_opt = pickle.load(f_in)
                os.remove(input_file)
                new_opt = deepcopy(opt)
                new_opt.update(job_in_opt)
                online_exp(new_opt, log_file)
                fout = open(output_file, 'w')
                fout.write('0\n')
                fout.close()
            except:
                fout = open(output_file, 'w')
                fout.write('Error in train: {}\n'.format(traceback.format_exc()))
                fout.close()
            log_print('job done {}'.format(job_num), log_file)

