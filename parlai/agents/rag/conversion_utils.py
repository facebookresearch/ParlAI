#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Conversion Scripts for RAG/DPR.
"""
from collections import OrderedDict
import os
import torch
from transformers import BertModel
from typing import Dict

from parlai.utils.io import PathManager
from parlai.utils.misc import recursive_getattr

# Mapping from BERT key to ParlAI Key
BERT_EMB_DICT_MAP = {
    'embeddings.word_embeddings': 'embeddings',
    'embeddings.position_embeddings': 'position_embeddings',
    'embeddings.token_type_embeddings': 'segment_embeddings',
    'embeddings.LayerNorm': 'norm_embeddings',
    'embeddings.dropout': 'dropout',
}


# List of weight keys not necessary in ParlAI Model
BERT_COMPATIBILITY_KEYS = ['embeddings.position_ids']


class BertConversionUtils:
    """
    Utilities for converting HFBertModels to ParlAI Models.
    """

    @staticmethod
    def load_bert_state(
        datapath: str,
        state_dict: Dict[str, torch.Tensor],
        pretrained_dpr_path: str,
        encoder_type: str = 'query',
    ) -> Dict[str, torch.Tensor]:
        """
        Load BERT State from HF Model, convert to ParlAI Model.

        :param state_dict:
            ParlAI model state_dict
        :param pretrained_dpr_path:
            path to pretrained DPR model
        :param encoder_type:
            whether we're loading a document or query encoder.

        :return new_state_dict:
            return a state_dict with loaded weights.
        """

        try:
            bert_model = BertModel.from_pretrained('bert-base-uncased')
        except OSError:
            model_path = PathManager.get_local_path(
                os.path.join(datapath, "bert_base_uncased")
            )
            bert_model = BertModel.from_pretrained(model_path)

        if pretrained_dpr_path:
            BertConversionUtils.load_dpr_model(
                bert_model, pretrained_dpr_path, encoder_type
            )
        bert_state_dict = bert_model.state_dict()
        for key in BERT_COMPATIBILITY_KEYS:
            bert_state_dict.pop(key, None)
        return_dict = BertConversionUtils.convert_bert_to_parlai(bert_state_dict)

        assert all(
            a in return_dict for a in state_dict
        ), f"not all weights are being loaded: {[k for k in state_dict if k not in return_dict]}"
        return return_dict

    @staticmethod
    def load_dpr_model(
        bert_model: BertModel, pretrained_dpr_path: str, encoder_type: str
    ):
        """
        Load saved state from pretrained DPR model directly into given bert_model.

        :param bert_model:
            bert model to load
        :param pretrained_dpr_path:
            path to pretrained DPR BERT Model
        :param encoder_type:
            whether we're loading a document or query encoder.
        """
        saved_state = torch.load(pretrained_dpr_path, map_location='cpu')
        model_to_load = (
            bert_model.module if hasattr(bert_model, 'module') else bert_model
        )

        prefix = 'question_model.' if encoder_type == 'query' else 'ctx_model.'
        prefix_len = len(prefix)
        encoder_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state['model_dict'].items()
            if key.startswith(prefix)
        }
        encoder_state.update(
            {k: v for k, v in saved_state['model_dict'].items() if 'encode_proj' in k}
        )
        try:
            model_to_load.load_state_dict(encoder_state)
        except RuntimeError:
            for key in BERT_COMPATIBILITY_KEYS:
                encoder_state[key] = recursive_getattr(model_to_load, key)
            model_to_load.load_state_dict(encoder_state)

    @staticmethod
    def convert_bert_to_parlai(
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a given BERT state dict to a ParlAI state dict.

        :param state_dict:
            BertModel state dict

        :return state_dict:
            return a state_dict that fits with ParlAI Models.
        """
        return_dict = OrderedDict()
        for each_key in state_dict.keys():
            mapped_key = each_key

            # 0. Skip pooler
            if 'pooler' in each_key:
                continue

            # 1. Map Embeddings
            for emb in BERT_EMB_DICT_MAP:
                mapped_key = mapped_key.replace(emb, BERT_EMB_DICT_MAP[emb])

            # 2. Map Layers
            if 'encoder' in each_key and 'layer' in each_key:
                mapped_key = mapped_key.replace('encoder.layer', 'layers')
            # 3. map attention
            if 'attention' in each_key:
                mapped_key = mapped_key.replace(
                    'attention.self.query', 'attention.q_lin'
                )
                mapped_key = mapped_key.replace('attention.self.key', 'attention.k_lin')
                mapped_key = mapped_key.replace(
                    'attention.self.value', 'attention.v_lin'
                )
                mapped_key = mapped_key.replace(
                    'attention.self.dropout', 'attention.attn_dropout'
                )
                mapped_key = mapped_key.replace(
                    'attention.output.dense', 'attention.out_lin'
                )
                mapped_key = mapped_key.replace('attention.output.LayerNorm', 'norm1')
                mapped_key = mapped_key.replace(
                    'attention.output.dropout', 'ffn.relu_dropout'
                )
            # 4. Map FFN
            if 'intermediate' in each_key:
                mapped_key = mapped_key.replace('intermediate.dense', 'ffn.lin1')
            if 'output' in each_key:
                mapped_key = mapped_key.replace('output.dense', 'ffn.lin2')
                mapped_key = mapped_key.replace('output.LayerNorm', 'norm2')
                mapped_key = mapped_key.replace('output.dropout', 'dropout')

            return_dict[mapped_key] = state_dict[each_key]

        return return_dict
