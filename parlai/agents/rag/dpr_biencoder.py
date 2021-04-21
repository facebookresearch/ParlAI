#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
DPR Bi-encoder.

Wraps a DPR Model within a ParlAI Bi-encoder.
"""
from parlai.agents.transformer.biencoder import BiencoderAgent
from parlai.agents.transformer.modules import TransformerMemNetModel
from parlai.agents.hugging_face.dict import HuggingFaceDictionaryAgent
from parlai.core.loader import register_agent
from parlai.core.opt import Opt
from parlai.core.torch_ranker_agent import TorchRankerAgent

from parlai.agents.rag.retrievers import RagDprQueryEncoder, RagDprDocumentEncoder

import torch
from transformers import BertConfig

try:
    from transformers import BertTokenizerFast as BertTokenizer
except ImportError:
    # No Rust Available
    from transformers import BertTokenizer

from typing import Dict


@register_agent('dpr_biencoder')
class DPRBiencoderAgent(BiencoderAgent):
    """
    Bi-encoder Wrapper for the DPR Models
    """

    def __init__(self, opt, shared=None):
        config: BertConfig = BertConfig.from_pretrained('bert-base-uncased')
        opt['n_heads'] = config.num_attention_heads
        opt['n_layers'] = config.num_hidden_layers
        opt['embedding_size'] = config.hidden_size
        opt['ffn_size'] = config.intermediate_size
        opt['dropout'] = config.hidden_dropout_prob
        opt['attention_dropout'] = config.attention_probs_dropout_prob
        opt['reduction_type'] = 'first'
        opt['n_positions'] = config.max_position_embeddings
        opt['activation'] = config.hidden_act
        opt['variant'] = 'xlm'
        opt['n_segments'] = config.type_vocab_size
        super().__init__(opt, shared)

    def build_model(self):
        return DPRBiencoderModel(self.opt, self.dict)

    def vectorize(self, *args, **kwargs):
        """
        Overrides BiencoderAgent to not add start/end.
        """
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        return TorchRankerAgent.vectorize(self, *args, **kwargs)

    def _set_text_vec(self, *args, **kwargs):
        """
        Overrides BiencoderAgent to not add start/end if init from DPR Rag Model.
        """
        return TorchRankerAgent._set_text_vec(self, *args, **kwargs)

    def vectorize_fixed_candidates(self, *args, **kwargs):
        """
        Vectorize fixed candidates.

        Overrides PolyEncoderAgent to not add start/end if init from DPR Rag Model
        """
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        return BiencoderAgent.vectorize_fixed_candidates(self, *args, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Overrides parent class to do nothing, as model is already loaded.
        """
        pass

    def build_dictionary(self):
        """
        Overrides TorchAgent.build_dictionary to possibly use a BERT Dictionary.
        """
        return BertTokenizerDictionaryAgent(self.opt)


class DPRBiencoderModel(TransformerMemNetModel):
    """
    Override TransformerMemNetModel to build DPR Model Components.
    """

    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)
        query_model = 'bert'
        document_model = 'bert'
        query_path = opt['model_file']
        document_path = opt['model_file']
        try:
            # determine if loading a RAG model
            loaded_opt = Opt.load(f"{query_path}.opt")
            if loaded_opt['model'] == 'internal:rag' and loaded_opt['query_model'] in [
                'bert',
                'bert_from_parlai_rag',
            ]:
                query_model = 'bert_from_parlai_rag'
            # document model is always frozen
            document_path = loaded_opt['dpr_model_file']

        except FileNotFoundError:
            pass
        self.context_encoder = RagDprQueryEncoder(
            opt, dpr_model=query_model, pretrained_path=query_path
        )
        self.cand_encoder = RagDprDocumentEncoder(
            opt, dpr_model=document_model, pretrained_path=document_path
        )


class BertTokenizerDictionaryAgent(HuggingFaceDictionaryAgent):
    def get_tokenizer(self, opt: Opt):
        return BertTokenizer.from_pretrained('bert-base-uncased')

    @property
    def add_special_tokens(self) -> bool:
        """
        Whether to add special tokens when tokenizing.
        """
        return True

    @property
    def skip_decode_special_tokens(self) -> bool:
        """
        Whether to decode special tokens when converting tokens to strings.
        """
        return True

    def override_special_tokens(self, opt: Opt):
        """
        Set special tokens according to bert tokenizer.
        """
        # now override
        self.start_token = self.hf_tokenizer.cls_token
        self.end_token = self.hf_tokenizer.sep_token
        self.null_token = self.hf_tokenizer.pad_token
        self.unk_token = self.hf_tokenizer.unk_token

        self._unk_token_idx = self.hf_tokenizer.unk_token_id

        self.start_idx = self[self.start_token]
        self.end_idx = self[self.end_token]
        self.null_idx = self[self.null_token]
