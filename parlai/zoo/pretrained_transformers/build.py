#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Pretrained model allowing to get the same performances as in
https://arxiv.org/abs/1905.01969
"""
from parlai.core.build_data import built, download_models, get_model_dir
import os
import os.path
import torch


def download(datapath):
    model_name = 'pretrained_transformers'
    mdir = os.path.join(get_model_dir(datapath), model_name)
    version = 'v1.0'
    models_built = built(mdir, version)
    if not models_built:
        opt = {'datapath': datapath}
        fnames = ['pretrained_transformers_v1.tar.gz']
        download_models(opt, fnames, model_name, version=version, use_model_type=False)
        print('Creating base models for bi and polyencoders')
        for pretrained_type in ['reddit', 'wikito']:
            path_cross = os.path.join(mdir, 'cross_model_huge_%s.mdl' % pretrained_type)
            path_bi = os.path.join(mdir, 'bi_model_huge_%s.mdl' % pretrained_type)
            path_poly = os.path.join(mdir, 'poly_model_huge_%s.mdl' % pretrained_type)
            create_bi_model(path_cross, path_bi)
            create_poly_model(path_cross, path_poly)


def create_bi_model(path_to_crossmodel, path_output):
    """ Create a biencoder model from a crossencoder model (that's to save
        space in the tar.gz)
    """
    loaded_model = torch.load(path_to_crossmodel)
    bi_model_params = {}
    for k, v in loaded_model['model'].items():
        if k == 'encoder.embeddings.weight':
            bi_model_params['embeddings.weight'] = v
            bi_model_params['cand_embeddings.weight'] = v
            bi_model_params['cand_encoder.embeddings.weight'] = v
            bi_model_params['context_encoder.embeddings.weight'] = v
            bi_model_params['memory_transformer.embeddings.weight'] = v
        elif k.startswith('encoder.'):
            bi_model_params[k.replace('encoder.', 'cand_encoder.')] = v
            bi_model_params[k.replace('encoder.', 'context_encoder.')] = v
            bi_model_params[k.replace('encoder.', 'memory_transformer.')] = v
    torch.save({'model': bi_model_params}, path_output)


def create_poly_model(path_to_crossmodel, path_output):
    """ Create a polyencoder model from a crossencoder model (that's to save
        space in the tar.gz)
    """
    loaded_model = torch.load(path_to_crossmodel)
    poly_model_params = {}
    for k, v in loaded_model['model'].items():
        if k == 'encoder.embeddings.weight':
            poly_model_params['encoder_cand.embeddings.weight'] = v
            poly_model_params['encoder_ctxt.embeddings.weight'] = v
        elif k.startswith('encoder.'):
            poly_model_params[k.replace('encoder.', 'encoder_cand.')] = v
            poly_model_params[k.replace('encoder.', 'encoder_ctxt.')] = v
    torch.save({'model': poly_model_params}, path_output)
