#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
DPR Bi-encoder.

Wraps a DPR Model within a ParlAI Bi-encoder.
"""
from parlai.agents.transformer.transformer import TransformerRankerAgent
from parlai.agents.transformer.modules import TransformerMemNetModel, TransformerEncoder
from parlai.agents.hugging_face.dict import HuggingFaceDictionaryAgent
from parlai.agents.rag.args import DPR_ZOO_MODEL
from parlai.agents.rag.conversion_utils import BertConversionUtils
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.loader import register_agent
from parlai.core.opt import Opt
from parlai.core.torch_ranker_agent import TorchRankerAgent
import parlai.utils.logging as logging


import os
import torch
import torch.cuda
import torch.nn
from transformers import BertConfig
from typing import Dict, Optional

try:
    from transformers import BertTokenizerFast as BertTokenizer
except ImportError:
    from transformers import BertTokenizer


@register_agent('dpr_agent')
class DPRAgent(TransformerRankerAgent):
    """
    TRA Wrapper for the DPR Models.
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
        return DPRModel(self.opt, self.dict)

    def vectorize(self, *args, **kwargs):
        """
        Overrides BiencoderAgent to not add start/end.
        """
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        return TorchRankerAgent.vectorize(self, *args, **kwargs)

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


class DprEncoder(TransformerEncoder):
    """
    Basically provide a wrapper around TransformerEncoder to load DPR Query/Document
    models.
    """

    def __init__(
        self,
        opt: Opt,
        dpr_model: str = 'bert',
        pretrained_path: str = DPR_ZOO_MODEL,
        encoder_type: str = 'query',
    ):
        # Override options
        config: BertConfig = BertConfig.from_pretrained('bert-base-uncased')
        pretrained_path = modelzoo_path(
            opt['datapath'], pretrained_path
        )  # type: ignore
        if not os.path.exists(pretrained_path):
            # when initializing from parlai rag models, the pretrained path
            # may not longer exist. This is fine if we've alread trained
            # the model.
            assert dpr_model == 'bert_from_parlai_rag'
            logging.error(f'Pretrained Path does not exist: {pretrained_path}')
            pretrained_path = modelzoo_path(
                opt['datapath'], DPR_ZOO_MODEL
            )  # type: ignore
            dpr_model = 'bert'
            logging.error(f'Setting to zoo model: {pretrained_path}')
        enc_opt = {
            "n_heads": config.num_attention_heads,
            "n_layers": config.num_hidden_layers,
            "embedding_size": config.hidden_size,
            "ffn_size": config.intermediate_size,
            "dropout": config.hidden_dropout_prob,
            "attention_dropout": config.attention_probs_dropout_prob,
            "activation": config.hidden_act,
            "variant": 'xlm',
            "reduction_type": 'first',
            "n_positions": config.max_position_embeddings,
            "n_segments": config.type_vocab_size,
        }
        embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        super().__init__(
            Opt(enc_opt),
            vocabulary_size=config.vocab_size,
            padding_idx=config.pad_token_id,
            embedding=embedding,
            reduction_type='first',
        )

        self._load_state(dpr_model, pretrained_path, encoder_type)

    def _load_state(self, dpr_model: str, pretrained_path: str, encoder_type: str):
        """
        Load pre-trained model states.

        :param dpr_model:
            which dpr model type we're using
        :param pretrained_path:
            path to pretrained model
        :param encoder_type:
            whether this is a query or document encoder
        """
        if dpr_model == 'bert':
            state_dict = BertConversionUtils.load_bert_state(
                self.state_dict(),
                pretrained_dpr_path=pretrained_path,
                encoder_type=encoder_type,
            )
            self.load_state_dict(state_dict)
        elif dpr_model == 'bert_from_parlai_rag':
            state_dict = torch.load(pretrained_path, map_location='cpu')["model"]
            key = f"{encoder_type}_encoder."
            state_dict = {
                k.split(key)[-1]: v for k, v in state_dict.items() if key in k
            }
            self.load_state_dict(state_dict)


class DprQueryEncoder(DprEncoder):
    """
    Query Encoder for DPR.
    """

    def __init__(self, *args, **kwargs):
        kwargs['encoder_type'] = 'query'
        super().__init__(*args, **kwargs)


class DprDocumentEncoder(DprEncoder):
    """
    Document Encoder for DPR.
    """

    def __init__(self, *args, **kwargs):
        kwargs['encoder_type'] = 'document'
        super().__init__(*args, **kwargs)


class DPRModel(TransformerMemNetModel):
    """
    Override TransformerMemNetModel to build DPR Model Components.
    """

    @classmethod
    def _get_build_options(cls, opt: Opt):
        """
        Return build options for DprEncoders.

        :return (query_model, query_path, document_model, document_path):
            query_model: dpr query model type
            query_path: path to pre-trained DPR model
            document_model: dpr document model type
            document_path: path to pre-trained document model
        """
        query_model = 'bert'
        document_model = 'bert'
        query_path = opt['model_file']
        document_path = opt['model_file']
        try:
            # determine if loading a RAG model
            loaded_opt = Opt.load(f"{query_path}.opt")
            document_path = loaded_opt.get('dpr_model_file', document_path)
            if loaded_opt['model'] in ['rag', 'fid'] and loaded_opt['query_model'] in [
                'bert',
                'bert_from_parlai_rag',
            ]:
                query_model = 'bert_from_parlai_rag'
                if loaded_opt['model'] == 'fid':
                    # document model is always frozen
                    # but may be loading a FiD-RAG Model
                    doc_loaded_opt = Opt.load(
                        f"{modelzoo_path(opt['datapath'], document_path)}.opt"
                    )
                    document_path = doc_loaded_opt.get('dpr_model_file', document_path)

        except FileNotFoundError:
            pass

        return query_model, query_path, document_model, document_path

    @classmethod
    def build_context_encoder(
        cls,
        opt: Opt,
        dictionary: DictionaryAgent,
        embedding: Optional[torch.nn.Embedding] = None,
        padding_idx: Optional[int] = None,
        reduction_type: str = 'mean',
    ):
        query_model, query_path, *_ = cls._get_build_options(opt)
        return DprQueryEncoder(opt, dpr_model=query_model, pretrained_path=query_path)

    @classmethod
    def build_candidate_encoder(
        cls,
        opt: Opt,
        dictionary: DictionaryAgent,
        embedding: Optional[torch.nn.Embedding] = None,
        padding_idx: Optional[int] = None,
        reduction_type: str = 'mean',
    ):
        _, _, document_model, document_path = cls._get_build_options(opt)
        return DprDocumentEncoder(
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
