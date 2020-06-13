#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Bi-encoder Agent.
"""
import torch
from .transformer import TransformerRankerAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent


class AddLabelFixedCandsTRA(TorchRankerAgent):
    """
    Override TorchRankerAgent to include label in fixed cands set.

    Necessary for certain IR tasks where given candidate set does not contain example
    labels.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.add_label_to_fixed_cands = opt.get('add_label_to_fixed_cands')
        if self.add_label_to_fixed_cands:
            self.ignore_bad_candidates = True

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Override to include new arg.
        """
        super(TorchRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('AddLabelFixedCandsTRA')
        agent.add_argument(
            '--add-label-to-fixed-cands',
            type='bool',
            default=True,
            hidden=True,
            help='When true, adds an example label to the fixed candidate set '
            'if not already present',
        )

    def _build_candidates(self, batch, source, mode):
        cands, cand_vecs, label_inds = super()._build_candidates(batch, source, mode)
        label_vecs = batch.label_vec  # [bsz] list of lists of LongTensors
        batchsize = (
            batch.text_vec.size(0)
            if batch.text_vec is not None
            else batch.image.size(0)
        )

        if source == 'fixed' and label_inds is None and self.add_label_to_fixed_cands:
            # Add label to fixed cands
            if label_vecs is not None:
                label_inds = label_vecs.new_empty((batchsize))
                for batch_idx, label_vec in enumerate(label_vecs):
                    max_c_len = cand_vecs.size(1)
                    label_vec_pad = label_vec.new_zeros(max_c_len).fill_(self.NULL_IDX)
                    if max_c_len < len(label_vec):
                        label_vec = label_vec[0:max_c_len]
                    label_vec_pad[0 : label_vec.size(0)] = label_vec
                    label_inds[batch_idx] = self._find_match(cand_vecs, label_vec_pad)
                    if label_inds[batch_idx] == -1:
                        cand_vecs = torch.cat((cand_vecs, label_vec_pad.unsqueeze(0)))
                        cands.append(batch.labels[batch_idx])
                        label_inds[batch_idx] = len(cands) - 1

        return (cands, cand_vecs, label_inds)

    def train_step(self, batch):
        """
        Override to clean up candidates.
        """
        output = super().train_step(batch)
        if self.candidates == 'fixed':
            self.fixed_candidates = self.fixed_candidates[: self.num_fixed_candidates]
        return output

    def eval_step(self, batch):
        """
        Override to clean up candidates.
        """
        output = super().eval_step(batch)
        if self.eval_candidates == 'fixed':
            self.fixed_candidates = self.fixed_candidates[: self.num_fixed_candidates]
        return output


class BiencoderAgent(TransformerRankerAgent):
    """
    Bi-encoder Transformer Agent.

    Equivalent of bert_ranker/biencoder but does not rely on an external library
    (hugging face).
    """

    def vectorize(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        obs = TorchRankerAgent.vectorize(self, *args, **kwargs)
        return obs

    def _vectorize_text(self, *args, **kwargs):
        """
        Override to add start end tokens.

        necessary for fixed cands.
        """
        if 'add_start' in kwargs:
            kwargs['add_start'] = True
            kwargs['add_end'] = True
        return super()._vectorize_text(*args, **kwargs)

    def _set_text_vec(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        obs = super()._set_text_vec(*args, **kwargs)
        if 'text_vec' in obs and 'added_start_end_tokens' not in obs:
            obs.force_set(
                'text_vec', self._add_start_end_tokens(obs['text_vec'], True, True)
            )
            obs['added_start_end_tokens'] = True
        return obs


class IRFriendlyBiencoderAgent(AddLabelFixedCandsTRA, BiencoderAgent):
    """
    Bi-encoder agent that allows for adding label to fixed cands.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add cmd line args.
        """
        AddLabelFixedCandsTRA.add_cmdline_args(argparser)
        BiencoderAgent.add_cmdline_args(argparser)
