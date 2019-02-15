#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy

import torch

from collections import OrderedDict

urls = {}
urls['dictionary'] = 'http://www.cs.toronto.edu/~rkiros/models/dictionary.txt'
urls['utable'] = 'http://www.cs.toronto.edu/~rkiros/models/utable.npy'
urls['uni_skip'] = 'http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz'


def load_dictionary(download_dir):
    path_dico = os.path.join(download_dir, 'dictionary.txt')
    if not os.path.exists(path_dico):
        os.system('mkdir -p ' + download_dir)
        os.system('wget {} -P {}'.format(urls['dictionary'], download_dir))
    with open(path_dico, 'r') as handle:
        dico_list = handle.readlines()
    dico = {word.strip(): idx for idx, word in enumerate(dico_list)}
    return dico


def load_emb_params(download_dir):
    table_name = 'utable'
    path_params = os.path.join(download_dir, table_name + '.npy')
    if not os.path.exists(path_params):
        os.system('mkdir -p ' + download_dir)
        os.system('wget {} -P {}'.format(urls[table_name], download_dir))
    params = numpy.load(path_params, encoding='latin1')  # to load from python2
    return params


def load_rnn_params(download_dir):
    skip_name = 'uni_skip'
    path_params = os.path.join(download_dir, skip_name + '.npz')
    if not os.path.exists(path_params):
        os.system('mkdir -p ' + download_dir)
        os.system('wget {} -P {}'.format(urls[skip_name], download_dir))
    params = numpy.load(path_params, encoding='latin1')  # to load from python2
    return params


def make_emb_state_dict(dictionary, parameters, vocab):
    weight = torch.zeros(len(vocab), 620)
    unknown_params = parameters[dictionary['UNK']]
    nb_unknown = 0
    for id_weight, word in enumerate(vocab):
        if word in dictionary:
            id_params = dictionary[word]
            params = parameters[id_params]
        else:
            # print('Warning: word `{}` not in dictionary'.format(word))
            params = unknown_params
            nb_unknown += 1
        weight[id_weight] = torch.from_numpy(params)
    state_dict = OrderedDict({'weight': weight})
    if nb_unknown > 0:
        print('Warning: {}/{} words are not in dictionary, thus set UNK'
              .format(nb_unknown, len(dictionary)))
    return state_dict


def make_gru_state_dict(p):
    s = OrderedDict()
    s['bias_ih_l0'] = torch.zeros(7200)
    s['bias_hh_l0'] = torch.zeros(7200)  # must stay equal to 0
    s['weight_ih_l0'] = torch.zeros(7200, 620)
    s['weight_hh_l0'] = torch.zeros(7200, 2400)
    s['weight_ih_l0'][:4800] = torch.from_numpy(p['encoder_W']).t()
    s['weight_ih_l0'][4800:] = torch.from_numpy(p['encoder_Wx']).t()
    s['bias_ih_l0'][:4800] = torch.from_numpy(p['encoder_b'])
    s['bias_ih_l0'][4800:] = torch.from_numpy(p['encoder_bx'])
    s['weight_hh_l0'][:4800] = torch.from_numpy(p['encoder_U']).t()
    s['weight_hh_l0'][4800:] = torch.from_numpy(p['encoder_Ux']).t()
    return s


def make_bayesian_state_dict(p):
    s = OrderedDict()
    s['gru_cell.weight_ir.weight'] = torch.from_numpy(p['encoder_W']).t()[:2400]
    s['gru_cell.weight_ii.weight'] = torch.from_numpy(p['encoder_W']).t()[2400:]
    s['gru_cell.weight_in.weight'] = torch.from_numpy(p['encoder_Wx']).t()

    s['gru_cell.weight_ir.bias'] = torch.from_numpy(p['encoder_b'])[:2400]
    s['gru_cell.weight_ii.bias'] = torch.from_numpy(p['encoder_b'])[2400:]
    s['gru_cell.weight_in.bias'] = torch.from_numpy(p['encoder_bx'])

    s['gru_cell.weight_hr.weight'] = torch.from_numpy(p['encoder_U']).t()[:2400]
    s['gru_cell.weight_hi.weight'] = torch.from_numpy(p['encoder_U']).t()[2400:]
    s['gru_cell.weight_hn.weight'] = torch.from_numpy(p['encoder_Ux']).t()
    return s
