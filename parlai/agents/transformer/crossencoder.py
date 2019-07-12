# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure -m transformer/generator works as expected
from .modules import TransformerEncoder
from .modules import get_n_positions_from_options
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .transformer import TransformerRankerAgent
import torch


class CrossencoderAgent(TorchRankerAgent):
    """ Equivalent of bert_ranker/crossencoder but does not rely on an external
        library (hugging face).
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)
        if self.use_cuda:
            self.rank_loss.cuda()
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            from parlai.core.distributed_utils import is_distributed

            if is_distributed():
                raise ValueError('Cannot combine --data-parallel and distributed mode')
            self.model = torch.nn.DataParallel(self.model)

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        TransformerRankerAgent.add_cmdline_args(argparser)
        return argparser

    def build_model(self, states=None):
        self.model = CrossEncoderModule(self.opt, self.dict, self.NULL_IDX)
        return self.model

    def vectorize(self, *args, **kwargs):
        """ Add the start and end token to the text.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        obs = super().vectorize(*args, **kwargs)
        return obs

    def _set_text_vec(self, *args, **kwargs):
        """ Add the start and end token to the text.
        """
        obs = super()._set_text_vec(*args, **kwargs)
        if 'text_vec' in obs:
            obs['text_vec'] = self._add_start_end_tokens(obs['text_vec'], True, True)
        return obs

    def concat_without_padding(self, text_idx, cand_idx, null_idx=0):
        """ if text_idx = [[1, 2, 3, 4, 0, 0  ]]
            and cand_idx = [[5, 6, 7, 8, 0, 0 ]]
            then result = (tokens, segments) where
            tokens = [[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]]
            segments = [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]
        """
        assert text_idx.size(0) == cand_idx.size(0)
        assert len(text_idx.size()) == 2
        assert len(cand_idx.size()) == 2
        segments_idx = [0, 1]
        text_idx = text_idx.cpu()
        cand_idx = cand_idx.cpu()
        cand_len = cand_idx.size(1)
        concat_len = text_idx.size(1) + cand_idx.size(1)
        tokens = text_idx.new_zeros(text_idx.size(0), concat_len) + null_idx
        segments = text_idx.new_zeros(text_idx.size(0), concat_len) + null_idx
        for i in range(len(tokens)):
            non_nuls = torch.sum(text_idx[i, :] != null_idx)
            tokens[i, 0:non_nuls] = text_idx[i, 0:non_nuls]
            segments[i, 0:non_nuls] = segments_idx[0]
            tokens[i, non_nuls : non_nuls + cand_len] = cand_idx[i, :]
            segments[i, non_nuls : non_nuls + cand_len] = segments_idx[1]
        if self.use_cuda:
            tokens = tokens.cuda()
            segments = segments.cuda()
        return tokens, segments

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
        tokens, segments = self.concat_without_padding(
            text_idx, cand_idx, self.NULL_IDX
        )
        scores = self.model(tokens, segments)
        scores = scores.view(bsz, num_cands_per_sample)
        return scores


class CrossEncoderModule(torch.nn.Module):
    """ A simple wrapper around the transformer encoder which adds a linear
        layer.
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
        """ Scores each concatenation text + candidate.
        """
        encoded = self.encoder(tokens, None, segments)
        res = self.linear_layer(encoded)
        return res
