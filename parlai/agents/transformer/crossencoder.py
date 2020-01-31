#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure -m transformer/generator works as expected
from .modules import TransformerEncoder
from .modules import get_n_positions_from_options
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .transformer import TransformerRankerAgent
from parlai.utils.torch import concat_without_padding
import torch


class CrossencoderAgent(TorchRankerAgent):
    """
    Equivalent of bert_ranker/crossencoder but does not rely on an external library
    (hugging face).
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            from parlai.utils.distributed import is_distributed

            if is_distributed():
                raise ValueError('Cannot combine --data-parallel and distributed mode')
            if shared is None:
                self.model = torch.nn.DataParallel(self.model)

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        TransformerRankerAgent.add_cmdline_args(argparser)
        argparser.set_defaults(encode_candidate_vecs=False)
        return argparser

    def build_model(self, states=None):
        return CrossEncoderModule(self.opt, self.dict, self.NULL_IDX)

    def vectorize(self, *args, **kwargs):
        """
        Add the start and end token to the text.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        obs = super().vectorize(*args, **kwargs)
        return obs

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

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        if cand_encs is not None:
            raise Exception(
                'Candidate pre-computation is impossible on the ' 'crossencoder'
            )
        num_cands_per_sample = cand_vecs.size(1)
        bsz = cand_vecs.size(0)
        text_idx = (
            batch.text_vec.unsqueeze(1)
            .expand(-1, num_cands_per_sample, -1)
            .contiguous()
            .view(num_cands_per_sample * bsz, -1)
        )
        cand_idx = cand_vecs.view(num_cands_per_sample * bsz, -1)
        tokens, segments = concat_without_padding(
            text_idx, cand_idx, self.use_cuda, self.NULL_IDX
        )
        scores = self.model(tokens, segments)
        scores = scores.view(bsz, num_cands_per_sample)
        return scores


class CrossEncoderModule(torch.nn.Module):
    """
    A simple wrapper around the transformer encoder which adds a linear layer.
    """

    def __init__(self, opt, dict, null_idx):
        super(CrossEncoderModule, self).__init__()
        n_positions = get_n_positions_from_options(opt)
        embeddings = torch.nn.Embedding(
            len(dict), opt['embedding_size'], padding_idx=null_idx
        )
        torch.nn.init.normal_(embeddings.weight, 0, opt['embedding_size'] ** -0.5)
        self.encoder = TransformerEncoder(
            n_heads=opt['n_heads'],
            n_layers=opt['n_layers'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dict),
            embedding=embeddings,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=null_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            reduction_type=opt.get('reduction_type', 'first'),
            n_positions=n_positions,
            n_segments=2,
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
        )
        self.linear_layer = torch.nn.Linear(opt['embedding_size'], 1)

    def forward(self, tokens, segments):
        """
        Scores each concatenation text + candidate.
        """
        encoded = self.encoder(tokens, None, segments)
        res = self.linear_layer(encoded)
        return res
