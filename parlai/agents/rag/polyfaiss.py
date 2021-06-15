#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
PolyFAISS Query Encoder.

Contains a Wrapper for the Dropout Poly-encoder for use as a RAG Query Encoder.
"""
import torch
import torch.cuda
import torch.nn

from typing import Tuple

from parlai.agents.transformer.polyencoder import PolyEncoderModule
from parlai.core.agents import create_agent
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
import parlai.utils.logging as logging

from parlai.agents.transformer.dropout_poly import DropoutPolyAgent, reduce_ctxt
from parlai.agents.rag.args import POLYENCODER_OPT_KEYS, TRANSFORMER_RANKER_BASE_OPT


class RagDropoutPolyWrapper(torch.nn.Module):
    """
    Wrapper for a DropoutPoly agent.

    Provides interface to RagModel as query/document encoder (separate), and provides
    interface to retrievers as a full poly-encoder.
    """

    def __init__(self, opt: Opt):
        super().__init__()
        self.model, self.dict = self._build_model(opt)
        model_file = modelzoo_path(opt['datapath'], opt['poly_faiss_model_file'])
        model_opt = Opt.load(f"{model_file}.opt")
        self.reduction_type = model_opt['poly_dropout_reduction_type']
        self.use_codes = model_opt.get('poly_dropout_use_codes', True)

    def _build_model(self, opt: Opt) -> Tuple[PolyEncoderModule, DictionaryAgent]:
        """
        Build poly-encoder module.

        :param opt:
            options from base RAG Model

        :return dropout poly-encoder:
            return dropout poly agent.
        """
        model_file = modelzoo_path(opt['datapath'], opt['poly_faiss_model_file'])
        model_opt = Opt.load(f'{model_file}.opt')

        create_model_opt = {
            **{k: model_opt[k] for k in TRANSFORMER_RANKER_BASE_OPT},
            **{k: model_opt[k] for k in POLYENCODER_OPT_KEYS},
            'model': 'transformer/dropout_poly',
            'init_model': model_file,
            'dict_file': f'{model_file}.dict',
            # necessary opt args
            'multitask_weights': [1],
            # dropout_poly args
            'poly_dropout_reduction_type': model_opt['poly_dropout_reduction_type'],
            'poly_dropout_use_codes': model_opt.get('poly_dropout_use_codes', True),
        }
        logging.disable()
        agent = create_agent(Opt(create_model_opt))
        logging.enable()
        assert isinstance(agent, DropoutPolyAgent)
        return agent.model, agent.dict

    def forward(self, query_vecs: torch.LongTensor) -> torch.Tensor:
        if self.use_codes:
            ctxt_rep, ctxt_mask, _ = self.model(ctxt_tokens=query_vecs)  # type: ignore
        else:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            old_type = model.type
            model.type = 'n_first'  # type: ignore

            ctxt_rep, ctxt_mask, _ = self.model(ctxt_tokens=query_vecs)  # type: ignore
            model.type = old_type  # type: ignore

        ctxt_rep = reduce_ctxt(ctxt_rep, ctxt_mask, self.reduction_type)
        return ctxt_rep

    @property
    def embedding_size(self) -> int:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        return model.encoder_ctxt.embedding_size  # type: ignore
