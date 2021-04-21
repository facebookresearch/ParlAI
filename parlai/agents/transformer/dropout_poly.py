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
from typing import Optional


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
        self.poly_dropout_reduction_type = opt['poly_dropout_reduction_type']
        self.poly_dropout_prob = opt['poly_dropout_prob']
        self.use_codes = opt.get('poly_dropout_use_codes', True)
        self.poly_type = opt['polyencoder_type']
        assert 0 <= self.poly_dropout_prob <= 1

        super().__init__(opt, shared)

    def score_candidates(
        self,
        batch: Batch,
        cand_vecs: torch.Tensor,
        cand_encs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score candidates.

        Essentially copies most of the PolyencoderAgent.score_candidates function,
        except for the final step.

        :param batch:
            batch to consider
        :param cand_vecs:
            tokenized candidate vectors
        :param cand_encs:
            optional tensor of encoded candidates

        :return score:
            return scores for each candidate for each batch sample
        """
        bsz = self._get_batch_size(batch)
        if self.use_codes:
            ctxt_rep, ctxt_rep_mask, _ = self.model(**self._model_context_input(batch))
        else:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.type = 'n_first'
            ctxt_rep, ctxt_rep_mask, _ = self.model(**self._model_context_input(batch))
            model.type = self.poly_type

        if cand_encs is not None:
            if bsz == 1:
                cand_rep = cand_encs
            else:
                cand_rep = cand_encs.expand(bsz, cand_encs.size(1), -1)
        # bsz x num cands x seq len
        elif len(cand_vecs.shape) == 3:
            _, _, cand_rep = self.model(cand_tokens=cand_vecs)
        # bsz x seq len (if batch cands) or num_cands x seq len (if fixed cands)
        elif len(cand_vecs.shape) == 2:
            _, _, cand_rep = self.model(cand_tokens=cand_vecs.unsqueeze(1))
            num_cands = cand_rep.size(0)  # will be bsz if using batch cands
            cand_rep = cand_rep.expand(num_cands, bsz, -1).transpose(0, 1).contiguous()
        else:
            raise RuntimeError('Cand vecs must be 2D or 3D')

        ### PERFORM DROPOUT ###
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
