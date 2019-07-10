# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

human_eval = {}

test_model = {
    'no_cuda': True,
    'model': 'legacy:seq2seq:0',
    'model_file': 'models:convai2/seq2seq/convai2_self_seq2seq_model',
    'dict_file': 'models:convai2/seq2seq/convai2_self_seq2seq_model.dict',
    'dict_lower': True,
    'batchsize': 1,
}

# (2)
baseline_model = {
    'no_cuda': True,
    'model_file': 'models:controllable_dialogue/convai2_finetuned_baseline',
    'beam_size': 20,
    'batchsize': 1,
    'beam_min_n_best': 10,
}

greedy_model = {
    'no_cuda': True,
    'model_file': 'models:controllable_dialogue/convai2_finetuned_baseline',
    'beam_size': 1,
    'batchsize': 1,
}

# (3)
pricing_test = {
    **baseline_model,
    'weighted_decoding': 'intrep_word:-1,intrep_2gram:-1,extrep_word:-1,extrep_2gram:-1',
}

# Repetition models round 1
repetition_model_setting05 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-0.5',
}

repetition_model_setting12 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-1.25',
}

repetition_model_setting35 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5',
}

repetition_model_settinginf = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-1e20',
}

repetition_model_setting35_settinginf = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
}

# NIWF INTERESTINGNESS MODELS
# We have two interestingness models.
# bfw means "beam feature weights" control method and ct means "conditional training" control method
# All the interestingness models have repetition control.

interesting_model_ct_setting0 = {
    'no_cuda': True,
    'model_file': 'models:controllable_dialogue/control_avgniwf10b10e',
    'beam_size': 20,
    'batchsize': 1,
    'beam_min_n_best': 10,
    'set_controls': 'avg_niwf:0',
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
}

interesting_model_ct_setting3 = {
    **interesting_model_ct_setting0,
    'set_controls': 'avg_niwf:3',
}

interesting_model_ct_setting5 = {
    **interesting_model_ct_setting0,
    'set_controls': 'avg_niwf:5',
}

interesting_model_ct_setting7 = {
    **interesting_model_ct_setting0,
    'set_controls': 'avg_niwf:7',
}

interesting_model_ct_setting9 = {
    **interesting_model_ct_setting0,
    'set_controls': 'avg_niwf:9',
}

# comparable to interesting_model_ct_setting0
interesting_model_bfw_setting200 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,niwf:-200',
}

# comparable to interesting_model_ct_setting3
interesting_model_bfw_setting075 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,niwf:0.75',
}

# comparable to interesting_model_ct_setting5
interesting_model_bfw_setting183 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,niwf:1.83',
}

# comparable to interesting_model_ct_setting7
interesting_model_bfw_setting242 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,niwf:2.42',
}

# comparable to interesting_model_ct_setting9
interesting_model_bfw_setting317 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,niwf:3.17',
}


# INQUISITIVENESS MODELS

inquisitive_model_ct_setting00 = {
    'no_cuda': True,
    'model_file': 'models:controllable_dialogue/control_questionb11e10',
    'beam_size': 20,
    'batchsize': 1,
    'beam_min_n_best': 10,
    'set_controls': 'question:0',
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
}

inquisitive_model_ct_setting01 = {
    **inquisitive_model_ct_setting00,
    'set_controls': 'question:1',
}

inquisitive_model_ct_setting04 = {
    **inquisitive_model_ct_setting00,
    'set_controls': 'question:4',
}

inquisitive_model_ct_setting07 = {
    **inquisitive_model_ct_setting00,
    'set_controls': 'question:7',
}

inquisitive_model_ct_setting10 = {
    **inquisitive_model_ct_setting00,
    'set_controls': 'question:10',
}

# Compared to inquisitive_model_ct_setting10, this removes extrep_2gram control
# (because it blocks questions), and adds beam reordering
# (i.e. given the top 10 candidates from beam search, choose the one which has lowest extrep_2gram).
# This should give much closer to 100% questions.
inquisitive_model_ct_setting10_better = {
    **inquisitive_model_ct_setting00,
    'set_controls': 'question:10',
    'weighted_decoding': 'extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
    'beam_reorder': 'best_extrep2gram_qn',
}

# RESPONSIVENESS MODELS

responsiveness_model_bfw_setting_minus_10 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,intrep_2gram:-1e20,lastuttsim:-10',
}

responsiveness_model_bfw_setting_00 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,intrep_2gram:-1e20',
}

responsiveness_model_bfw_setting_05 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,intrep_2gram:-1e20,lastuttsim:5',
}

responsiveness_model_bfw_setting_10 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,intrep_2gram:-1e20,lastuttsim:10',
}

responsiveness_model_bfw_setting_13 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,intrep_2gram:-1e20,lastuttsim:13',
}


# NIDF CT INTERESTINGNESS MODELS
# CT buckets 0,2,4,7,9

interesting_nidf_model_ct_setting0 = {
    'no_cuda': True,
    'model_file': 'models:controllable_dialogue/control_avgnidf10b10e',
    'beam_size': 20,
    'batchsize': 1,
    'beam_min_n_best': 10,
    'set_controls': 'avg_nidf:0',
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
}

interesting_nidf_model_ct_setting2 = {
    **interesting_nidf_model_ct_setting0,
    'set_controls': 'avg_nidf:2',
}

interesting_nidf_model_ct_setting4 = {
    **interesting_nidf_model_ct_setting0,
    'set_controls': 'avg_nidf:4',
}

interesting_nidf_model_ct_setting7 = {
    **interesting_nidf_model_ct_setting0,
    'set_controls': 'avg_nidf:7',
}

interesting_nidf_model_ct_setting9 = {
    **interesting_nidf_model_ct_setting0,
    'set_controls': 'avg_nidf:9',
}


# BFW NIDF INTERESTINGNESS MODELS
# weights -10,-4,4,6,8 (0 is same as repetition_model_setting35_settinginf)

interesting_nidf_model_bfw_setting_minus_10 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:-10',
}

interesting_nidf_model_bfw_setting_minus_04 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:-4',
}

interesting_nidf_model_bfw_setting_04 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:4',
}

interesting_nidf_model_bfw_setting_06 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:6',
}

interesting_nidf_model_bfw_setting_08 = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:8',
}
