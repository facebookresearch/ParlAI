#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Dropout Poly-encoder.

The dropout poly-encoder mimics a combination of a bi-encoder & poly-encoder during
training, and during inference is exactly a poly-encoder.
"""
from parlai.core.params import ParlaiParser
from parlai.agents.transformer.polyencoder import PolyencoderAgent
from parlai.core.opt import Opt
from parlai.core.torch_agent import Batch

import random
import torch
from typing import Optional, Tuple


def reduce_ctxt(
    ctxt: torch.Tensor, mask: torch.Tensor, reduction_type: str
) -> torch.Tensor:
    """
    Reduce ctxt tensor to appropriate dimension for bi-encoder scoring.

    Functionality copied from TransformerEncoder.reduce_output.

    :param ctxt:
        all transformer outputs.
    :param mask:
        context mask
    :param reduction_type:
        how to reduce the context

    :return reduced:
        return reduced context tensor
    """
    if reduction_type == 'first':
        return ctxt[:, 0, :]
    elif reduction_type == 'max':
        return ctxt.max(dim=1)[0]
    elif reduction_type == 'mean':
        divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(ctxt)
        output = ctxt.sum(dim=1) / divisor
        return output
    else:
        raise ValueError("Can't handle --reduction-type {}".format(reduction_type))


class DropoutPolyAgent(PolyencoderAgent):
    """
    Dropout Polyencoder Agent.

    Overrides score_candidates to, sometimes, score candidates in the bi-encoder method.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group('Dropout Polyencoder Arguments')
        agent.add_argument(
            '--poly-dropout-prob',
            type=float,
            default=0.5,
            help='How often to switch to a bi-encoder setup during training',
        )
        agent.add_argument(
            '--poly-dropout-reduction-type',
            type=str,
            choices=['first', 'max', 'mean'],
            default='first',
            help='how to reduce output when ignoring poly-codes during training',
        )
        agent.add_argument(
            '--poly-dropout-use-codes',
            type='bool',
            default=True,
            help='Whether to attend over codes prior to dropout method.',
        )

        return parser

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.poly_dropout_reduction_type = opt['poly_dropout_reduction_type']
        self.poly_dropout_prob = opt['poly_dropout_prob']
        self.use_codes = opt.get('poly_dropout_use_codes', True)
        assert 0 <= self.poly_dropout_prob <= 1

    def get_ctxt_rep(self, batch: Batch) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Encode context representation.
        """
        if self.use_codes:
            ctxt_rep, ctxt_rep_mask, _ = self.model(**self._model_context_input(batch))
        else:
            # In dataparallel, the model is the `model.module`
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.type = 'n_first'
            ctxt_rep, ctxt_rep_mask, _ = self.model(**self._model_context_input(batch))
            model.type = self.opt['polyencoder_type']

        return ctxt_rep, ctxt_rep_mask

    def get_scores(
        self,
        ctxt_rep: torch.Tensor,
        ctxt_rep_mask: torch.BoolTensor,
        cand_rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score context with candidates.

        During training, with some probability we fall back to the Bi-encoder scoring
        method.
        """
        if self.is_training and random.random() < self.poly_dropout_prob:
            ctxt_rep = reduce_ctxt(
                ctxt_rep, ctxt_rep_mask, self.poly_dropout_reduction_type
            )
            scores = torch.bmm(ctxt_rep.unsqueeze(1), cand_rep.transpose(1, 2)).squeeze(
                1
            )
        else:
            scores = self.model(
                ctxt_rep=ctxt_rep, ctxt_rep_mask=ctxt_rep_mask, cand_rep=cand_rep
            )
        return scores
