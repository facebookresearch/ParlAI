# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure -m transformer/generator works as expected
from .modules import TransformerEncoder
from .modules import get_n_positions_from_options
from .modules import surround
from parlai.core.agents import Agent
from parlai.core.utils import warn_once
from parlai.core.utils import padded_3d
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .transformer import TransformerRankerAgent
from .modules import BasicAttention, MultiHeadAttention
import torch

class PolyencoderAgent(TorchRankerAgent):
    """ Equivalent of bert_ranker/polyencoder and biencoder_multiple_output
        but does not rely on an external library (hugging face).
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        TransformerRankerAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Cross Arguments')
        agent.add_argument('--polyencoder-type', type=str, default='codes',
                           choices=['codes', 'first_n'],
                           help='Type of polyencoder, either we compute'
                                'vectors using codes + attention, or we '
                                'simply take the first N vectors.')
        agent.add_argument('--poly-n-codes', type=int, default=64,
                           help='number of vectors used to represent the context'
                                'in the case of first_n, those are the number'
                                'of vectors that are considered.')
        agent.add_argument('--poly-attention-type', type=str, default='basic',
                           choices=['basic', 'basic_sqrt', 'multihead'],
                           help='Type of the top aggregation layer of the poly-'
                                'encoder (where the candidate representation is'
                                'the key)')
        agent.add_argument('--poly-attention-num-heads', type=int, default=4,
                           help='In case poly-attention-type is multihead, '
                                'specify the number of heads')

        # Those arguments are here in case where polyencoder type is 'code'
        agent.add_argument('--codes-attention-type', type=str, default='basic',
                           choices=['basic', 'basic_sqrt', 'multihead'],
                           help='Type ')
        agent.add_argument('--codes-attention-num-heads', type=int, default=4,
                           help='In case codes-attention-type is multihead, '
                                'specify the number of heads')
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)
        if self.use_cuda:
            self.rank_loss.cuda()
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            from parlai.core.distributed_utils import is_distributed
            if is_distributed():
                raise ValueError(
                    'Cannot combine --data-parallel and distributed mode'
                )
            self.model = torch.nn.DataParallel(self.model)


    def build_model(self, states=None):
        self.model = CrossEncoderModule(self.opt, self.dict, self.NULL_IDX)
        return self.model

    def vectorize(self, *args, **kwargs):
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        obs = super().vectorize(*args, **kwargs)
        return obs

    def _set_text_vec(self, *args, **kwargs):
        """ Add start and end tokens for set_text_vec()
        """
        obs = super()._set_text_vec(*args, **kwargs)
        if 'text_vec' in obs:
            obs['text_vec'] = surround(obs['text_vec'],
                                       self.START_IDX,
                                       self.END_IDX)
        return obs

    def concat_without_padding(self, text_idx, cand_idx,
                               null_idx= 0, segments_idx=[0, 1]):
        """ if text_idx = [[1, 2, 3, 4, 0, 0  ]]
            and cand_idx = [[5, 6, 7, 8, 0, 0 ]]
            then result = (tokens, segments) where
            tokens = [[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]]
            segments = [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]
        """
        assert text_idx.size(0) == cand_idx.size(0)
        assert len(text_idx.size()) == 2
        assert len(cand_idx.size()) == 2
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
            tokens[i, non_nuls:non_nuls+cand_len] = cand_idx[i, :]
            segments[i, non_nuls:non_nuls+cand_len] = segments_idx[1]
        if self.use_cuda:
            tokens = tokens.cuda()
            segments = segments.cuda()
        return tokens, segments


    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        num_cands_per_sample = cand_vecs.size(1)
        bsz =  cand_vecs.size(0)
        text_idx = (batch.text_vec.unsqueeze(1)
                                  .expand(-1, num_cands_per_sample, -1)
                                  .contiguous()
                                  .view(num_cands_per_sample * bsz, -1))
        cand_idx = cand_vecs.view(num_cands_per_sample * bsz, -1)
        tokens, segments = self.concat_without_padding(text_idx, cand_idx,
                                                       self.NULL_IDX)
        scores = self.model(tokens, segments)
        scores = scores.view(bsz, num_cands_per_sample)
        return scores

class PolyEncoderModule(torch.nn.Module):
    """ Allow to reproduce the experiments from the polypaper.
    """

    def __init__(self, opt, dict, null_idx):
        super(CrossEncoderModule, self).__init__()
        self.null_idx = null_idx
        self.encoder_context = get_encoder(opt, dict, null_idx, 'none')
        self.encoder_cand = get_encoder(opt, dict, null_idx, 'mean')

        self.type = opt['polyencoder_type']
        self.n_codes = opt['poly_n_codes']
        self.attention_type = opt['poly_attention_type']
        self.attention_num_heads = opt['poly_attention_num_heads']
        self.codes_attention_type = opt['codes_attention_type']
        self.codes_attention_num_heads = opt['codes_attention_num_heads']
        embed_dim = opt['embedding_size']

        # In case it's a polyencoder with code.
        if self.type == 'codes':
            # experimentally it seems that random with size = 1 was good.
            codes = torch.empty(self.n_codes,embed_dim))
            codes = torch.nn.init.uniform_(codes)
            self.codes = torch.nn.Parameter(codes)

            # The attention for the codes.
            if self.codes_attention_type == 'multihead':
                self.code_attention = MultiHeadAttention(
                                        self.codes_attention_num_heads,
                                        embed_dim,
                                        opt['dropout'])
            elif self.codes_attention_type == 'basic_sqrt':
                self.code_attention = BasicAttention(dim=1, attn='sqrt')
            elif self.codes_attention_type == 'basic':
                self.code_attention = BasicAttention(dim=1, attn='basic')

        # The final attention (the one that takes the candidate as key)
        if self.attention_type == 'multihead':
            self.attention = MultiHeadAttention(
                                self.attention_num_heads,
                                opt['embedding_size'],
                                opt['dropout'])
        elif self.attention_type == 'basic_sqrt':
            self.attention = BasicAttention(dim=1, attn='sqrt')
        elif self.attention_type == 'basic':
            self.attention = BasicAttention(dim=1, attn='basic')




    def get_encoder(self, opt, dict, null_idx, reduction_type):
        n_positions = get_n_positions_from_options(opt)
        embeddings = torch.nn.Embedding(
            len(dict),
            opt['embedding_size'],
            padding_idx=null_idx
        )
        torch.nn.init.normal_(embeddings.weight, 0,
                              opt['embedding_size'] ** -0.5)
        return TransformerEncoder(
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
            reduction_type= reduction_type,
            n_positions=n_positions,
            n_segments=2,
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'])

    def encode(self,tokens_context, tokens_candidate):
        """
            :param tokens_context:
                2D long tensor, batchsize x sent_len
            :param tokens_candidate:
                2D long tensor, batchsize x sent_len
                Note this is 2D, if you need to perform operation on 3D, please
                do it before hand
            :returns: (ctxt_rep, ctxt_mask, cand_rep)
                - ctxt_rep 3D float tensor, batchsize x n_codes x dim
                - ctxt_mask byte:  batchsize x n_codes (all 1 in case
                    of polyencoder with code. Which are the vectors to use
                    in the ctxt_rep)
                - cand_rep (2D float tensor) batchsize x dim
        """
        cand_embed = self.encoder_cand(tokens_candidate)
        # get context_representation. Now that depends on the cases.
        bs = context_output.size(0)
        dim = context_output.size(2)
        ctxt_out, ctxt_mask = self.encoder_context(tokens_context)
        if self.attention_type == 'codes'
            # Basic Attention and MultiHeadAttention share the same API.
            ctxt_rep = self.code_attention(self.codes.repeat(bs,1,1),
                                           ctxt_out,
                                           ctxt_mask)
            ctxt_rep_mask = ctxt_rep.new_ones(bs, self.n_codes).byte()

        elif self.attention_type == 'n_first':
            # Expand the output if it is not long enough
            if ctxt_out.size(1) < self.n_codes:
                difference = self.n_codes - ctxt_out.size(1)
                extra_rep = ctxt_out.new_zeros(bs, difference, dim)
                ctxt_rep = torch.cat([ctxt_out, extra_rep], dim=1)
                extra_mask = ctxt_mask.new_zeros(bs, difference)
                ctxt_rep_mask = torch.cat([ctxt_mask, extra_mask], dim=1)
            else:
                ctxt_rep = ctxt_out[:, 0:self.n_codes, :]
                ctxt_rep_mask = ctxt_mask[:, 0:self.n_codes]

        return ctxt_rep, ctxt_rep_mask, cand_embed





    def forward(self, operation_type, tokens_context, tokens_candidate):
        """ Due to a limitation of parlai, we have to have one single model
            in the agent. And because we want to be able to use data-parallel,
            we need to have one single forward() method.
            Therefore the operation_type can be either 'encode' or 'score'.
        """
        if operation_type == 'encode':
            return self.encode(tokens_context, tokens_candidate)
        elif operation_type == 'score':
            return self.score()
        raise Exception('Unsupported operation: %s' % operation_type)
