#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure -m transformer/generator works as expected
"""
Poly-encoder Agent.
"""
from .biencoder import AddLabelFixedCandsTRA
from .modules import TransformerEncoder
from .modules import get_n_positions_from_options
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .transformer import TransformerRankerAgent
from .modules import BasicAttention, MultiHeadAttention
import torch


class PolyencoderAgent(TorchRankerAgent):
    """
    Poly-encoder Agent.

    Equivalent of bert_ranker/polyencoder and biencoder_multiple_output but does not
    rely on an external library (hugging face).
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        TransformerRankerAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Polyencoder Arguments')
        agent.add_argument(
            '--polyencoder-type',
            type=str,
            default='codes',
            choices=['codes', 'n_first'],
            help='Type of polyencoder, either we compute'
            'vectors using codes + attention, or we '
            'simply take the first N vectors.',
            recommended='codes',
        )
        agent.add_argument(
            '--poly-n-codes',
            type=int,
            default=64,
            help='number of vectors used to represent the context'
            'in the case of n_first, those are the number'
            'of vectors that are considered.',
            recommended=64,
        )
        agent.add_argument(
            '--poly-attention-type',
            type=str,
            default='basic',
            choices=['basic', 'sqrt', 'multihead'],
            help='Type of the top aggregation layer of the poly-'
            'encoder (where the candidate representation is'
            'the key)',
            recommended='basic',
        )
        agent.add_argument(
            '--polyencoder-attention-keys',
            type=str,
            default='context',
            choices=['context', 'position'],
            help='Input emb vectors for the first level of attention. '
            'Context refers to the context outputs; position refers to the '
            'computed position embeddings.',
            recommended='context',
        )
        agent.add_argument(
            '--poly-attention-num-heads',
            type=int,
            default=4,
            help='In case poly-attention-type is multihead, '
            'specify the number of heads',
        )

        # Those arguments are here in case where polyencoder type is 'code'
        agent.add_argument(
            '--codes-attention-type',
            type=str,
            default='basic',
            choices=['basic', 'sqrt', 'multihead'],
            help='Type ',
            recommended='basic',
        )
        agent.add_argument(
            '--codes-attention-num-heads',
            type=int,
            default=4,
            help='In case codes-attention-type is multihead, '
            'specify the number of heads',
        )
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)
        if self.use_cuda:
            self.rank_loss.cuda()
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            from parlai.utils.distributed import is_distributed

            if is_distributed():
                raise ValueError('Cannot combine --data-parallel and distributed mode')
            if shared is None:
                self.model = torch.nn.DataParallel(self.model)

    def build_model(self, states=None):
        """
        Return built model.
        """
        return PolyEncoderModule(self.opt, self.dict, self.NULL_IDX)

    def vectorize(self, *args, **kwargs):
        """
        Add the start and end token to the labels.
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

    def vectorize_fixed_candidates(self, *args, **kwargs):
        """
        Vectorize fixed candidates.

        Override to add start and end token when computing the candidate encodings in
        interactive mode.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        return super().vectorize_fixed_candidates(*args, **kwargs)

    def _make_candidate_encs(self, vecs):
        """
        Make candidate encs.

        The polyencoder module expects cand vecs to be 3D while torch_ranker_agent
        expects it to be 2D. This requires a little adjustment (used in interactive mode
        only)
        """
        rep = super()._make_candidate_encs(vecs)
        return rep.transpose(0, 1).contiguous()

    def encode_candidates(self, padded_cands):
        """
        Encode candidates.
        """
        padded_cands = padded_cands.unsqueeze(1)
        _, _, _, cand_rep = self.model(cand_tokens=padded_cands)
        return cand_rep

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        Score candidates.

        The Poly-encoder encodes the candidate and context independently. Then, the
        model applies additional attention before ultimately scoring a candidate.
        """
        bsz = batch.text_vec.size(0)
        ctxt_rep, ctxt_rep_mask, ctxt_pos, _ = self.model(ctxt_tokens=batch.text_vec)

        if cand_encs is not None:
            if bsz == 1:
                cand_rep = cand_encs
            else:
                cand_rep = cand_encs.expand(bsz, cand_encs.size(1), -1)
        # bsz x num cands x seq len
        elif len(cand_vecs.shape) == 3:
            _, _, _, cand_rep = self.model(cand_tokens=cand_vecs)
        # bsz x seq len (if batch cands) or num_cands x seq len (if fixed cands)
        elif len(cand_vecs.shape) == 2:
            _, _, _, cand_rep = self.model(cand_tokens=cand_vecs.unsqueeze(1))
            num_cands = cand_rep.size(0)  # will be bsz if using batch cands
            cand_rep = cand_rep.expand(num_cands, bsz, -1).transpose(0, 1).contiguous()

        scores = self.model(
            ctxt_rep=ctxt_rep,
            ctxt_rep_mask=ctxt_rep_mask,
            cand_rep=cand_rep,
            ctxt_pos=ctxt_pos,
        )
        return scores

    def load_state_dict(self, state_dict):
        """
        Override to account for codes.
        """
        if self.model.type == 'codes' and 'codes' not in state_dict:
            state_dict['codes'] = self.model.codes
        super().load_state_dict(state_dict)


class PolyEncoderModule(torch.nn.Module):
    """
    Poly-encoder model.

    See https://arxiv.org/abs/1905.01969 for more details
    """

    def __init__(self, opt, dict, null_idx):
        super(PolyEncoderModule, self).__init__()
        self.null_idx = null_idx
        self.encoder_ctxt = self.get_encoder(opt, dict, null_idx, 'none_with_pos_embs')
        self.encoder_cand = self.get_encoder(opt, dict, null_idx, opt['reduction_type'])

        self.type = opt['polyencoder_type']
        self.n_codes = opt['poly_n_codes']
        self.attention_type = opt['poly_attention_type']
        self.attention_keys = opt.get('polyencoder_attention_keys', 'context')
        self.attention_num_heads = opt['poly_attention_num_heads']
        self.codes_attention_type = opt['codes_attention_type']
        self.codes_attention_num_heads = opt['codes_attention_num_heads']
        embed_dim = opt['embedding_size']

        # In case it's a polyencoder with code.
        if self.type == 'codes':
            # experimentally it seems that random with size = 1 was good.
            codes = torch.empty(self.n_codes, embed_dim)
            codes = torch.nn.init.uniform_(codes)
            self.codes = torch.nn.Parameter(codes)

            # The attention for the codes.
            if self.codes_attention_type == 'multihead':
                self.code_attention = MultiHeadAttention(
                    self.codes_attention_num_heads, embed_dim, opt['dropout']
                )
            elif self.codes_attention_type == 'sqrt':
                self.code_attention = PolyBasicAttention(
                    self.type, self.n_codes, dim=2, attn='sqrt', get_weights=False
                )
            elif self.codes_attention_type == 'basic':
                self.code_attention = PolyBasicAttention(
                    self.type, self.n_codes, dim=2, attn='basic', get_weights=False
                )

        # The final attention (the one that takes the candidate as key)
        if self.attention_type == 'multihead':
            self.attention = MultiHeadAttention(
                self.attention_num_heads, opt['embedding_size'], opt['dropout']
            )
        else:
            self.attention = PolyBasicAttention(
                self.type,
                self.n_codes,
                dim=2,
                attn=self.attention_type,
                get_weights=False,
            )

    def get_encoder(self, opt, dict, null_idx, reduction_type):
        """
        Return encoder, given options.

        :param opt:
            opt dict
        :param dict:
            dictionary agent
        :param null_idx:
            null/pad index into dict
        :reduction_type:
            reduction type for the encoder

        :return:
            a TransformerEncoder, initialized correctly
        """
        n_positions = get_n_positions_from_options(opt)
        embeddings = torch.nn.Embedding(
            len(dict), opt['embedding_size'], padding_idx=null_idx
        )
        torch.nn.init.normal_(embeddings.weight, 0, opt['embedding_size'] ** -0.5)
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
            reduction_type=reduction_type,
            n_positions=n_positions,
            n_segments=2,
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
        )

    def attend(self, attention_layer, queries, keys, values, mask):
        """
        Apply attention.

        :param attention_layer:
            nn.Module attention layer to use for the attention
        :param queries:
            the queries for attention
        :param keys:
            the keys for attention
        :param values:
            the values for attention
        :param mask:
            mask for the attention keys

        :return:
            the result of applying attention to the values, with weights computed
            wrt to the queries and keys.
        """
        if keys is None:
            keys = values
        if isinstance(attention_layer, PolyBasicAttention):
            return attention_layer(queries, keys, mask_ys=mask, values=values)
        elif isinstance(attention_layer, MultiHeadAttention):
            return attention_layer(queries, keys, values, mask)
        else:
            raise Exception('Unrecognized type of attention')

    def encode(self, ctxt_tokens, cand_tokens):
        """
        Encode a text sequence.

        :param ctxt_tokens:
            2D long tensor, batchsize x sent_len
        :param cand_tokens:
            3D long tensor, batchsize x num_cands x sent_len
            Note this will actually view it as a 2D tensor
        :return:
            (ctxt_rep, ctxt_mask, ctxt_pos, cand_rep)
            - ctxt_rep 3D float tensor, batchsize x n_codes x dim
            - ctxt_mask byte:  batchsize x n_codes (all 1 in case
            of polyencoder with code. Which are the vectors to use
            in the ctxt_rep)
            - ctxt_pos 3D float tensor, batchsize x sent_len x dim
            - cand_rep (3D float tensor) batchsize x num_cands x dim
        """
        cand_embed = None
        ctxt_rep = None
        ctxt_rep_mask = None
        ctxt_pos = None
        if cand_tokens is not None:
            assert len(cand_tokens.shape) == 3
            bsz = cand_tokens.size(0)
            num_cands = cand_tokens.size(1)
            cand_embed = self.encoder_cand(cand_tokens.view(bsz * num_cands, -1))
            cand_embed = cand_embed.view(bsz, num_cands, -1)

        if ctxt_tokens is not None:
            assert len(ctxt_tokens.shape) == 2
            bsz = ctxt_tokens.size(0)
            # get context_representation. Now that depends on the cases.
            ctxt_out, ctxt_mask, ctxt_pos = self.encoder_ctxt(ctxt_tokens)
            att_keys = ctxt_out if self.attention_keys == 'context' else ctxt_pos
            dim = ctxt_out.size(2)

            if self.type == 'codes':
                ctxt_rep = self.attend(
                    self.code_attention,
                    queries=self.codes.repeat(bsz, 1, 1),
                    keys=att_keys,
                    values=ctxt_out,
                    mask=ctxt_mask,
                )
                ctxt_pos = None  # we don't need this anymore
                ctxt_rep_mask = ctxt_rep.new_ones(bsz, self.n_codes).byte()

            elif self.type == 'n_first':
                # Expand the output if it is not long enough
                if ctxt_out.size(1) < self.n_codes:
                    difference = self.n_codes - ctxt_out.size(1)
                    extra_rep = ctxt_out.new_zeros(bsz, difference, dim)
                    ctxt_rep = torch.cat([ctxt_out, extra_rep], dim=1)
                    ctxt_pos = torch.cat([ctxt_pos, extra_rep], dim=1)
                    extra_mask = ctxt_mask.new_zeros(bsz, difference)
                    ctxt_rep_mask = torch.cat([ctxt_mask, extra_mask], dim=1)
                else:
                    ctxt_rep = ctxt_out[:, 0 : self.n_codes, :]
                    ctxt_pos = ctxt_pos[:, 0 : self.n_codes, :]
                    ctxt_rep_mask = ctxt_mask[:, 0 : self.n_codes]

        return ctxt_rep, ctxt_rep_mask, ctxt_pos, cand_embed

    def score(self, ctxt_rep, ctxt_rep_mask, ctxt_pos, cand_embed):
        """
        Score the candidates.

        :param ctxt_rep:
            3D float tensor, bsz x ctxt_len x dim
        :param ctxt_rep_mask:
            2D byte tensor, bsz x ctxt_len, in case there are some elements
            of the ctxt that we should not take into account.
        :param ctx_pos: 3D float tensor, bsz x sent_len x dim
        :param cand_embed: 3D float tensor, bsz x num_cands x dim

        :return: scores, 2D float tensor: bsz x num_cands
        """
        # Attention keys determined by self.attention_keys
        # 'context' == use context final rep; otherwise use context position embs
        keys = ctxt_rep if self.attention_keys == 'context' else ctxt_pos
        # reduces the context representation to a 3D tensor bsz x num_cands x dim
        ctxt_final_rep = self.attend(
            self.attention, cand_embed, keys, ctxt_rep, ctxt_rep_mask
        )
        scores = torch.sum(ctxt_final_rep * cand_embed, 2)
        return scores

    def forward(
        self,
        ctxt_tokens=None,
        cand_tokens=None,
        ctxt_rep=None,
        ctxt_rep_mask=None,
        ctxt_pos=None,
        cand_rep=None,
    ):
        """
        Forward pass of the model.

        Due to a limitation of parlai, we have to have one single model
        in the agent. And because we want to be able to use data-parallel,
        we need to have one single forward() method.
        Therefore the operation_type can be either 'encode' or 'score'.

        :param ctxt_tokens:
            tokenized contexts
        :param cand_tokens:
            tokenized candidates
        :param ctxt_rep:
            (bsz x num_codes x hsz)
            encoded representation of the context. If self.type == 'codes', these
            are the context codes. Otherwise, they are the outputs from the
            encoder
        :param ctxt_rep_mask:
            mask for ctxt rep
        :param ctxt_pos:
            position embeddings for the ctxt_rep. If self.type == 'codes', these
            are None, as their use is earlier in the pipeline.
        :param cand_rep:
            encoded representation of the candidates
        """
        if ctxt_tokens is not None or cand_tokens is not None:
            return self.encode(ctxt_tokens, cand_tokens)
        elif (
            ctxt_rep is not None and ctxt_rep_mask is not None and cand_rep is not None
        ):
            # ctxt_pos can be none, if we are using codes (not first M)
            return self.score(ctxt_rep, ctxt_rep_mask, ctxt_pos, cand_rep)
        raise Exception('Unsupported operation')


class PolyBasicAttention(BasicAttention):
    """
    Override basic attention to account for edge case for polyencoder.
    """

    def __init__(self, poly_type, n_codes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poly_type = poly_type
        self.n_codes = n_codes

    def forward(self, *args, **kwargs):
        """
        Forward pass.

        Account for accidental dimensionality reduction when num_codes is 1 and the
        polyencoder type is 'codes'
        """
        lhs_emb = super().forward(*args, **kwargs)
        if self.poly_type == 'codes' and self.n_codes == 1 and len(lhs_emb.shape) == 2:
            lhs_emb = lhs_emb.unsqueeze(self.dim - 1)
        return lhs_emb


class IRFriendlyPolyencoderAgent(AddLabelFixedCandsTRA, PolyencoderAgent):
    """
    Poly-encoder agent that allows for adding label to fixed cands.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add cmd line args.
        """
        super(AddLabelFixedCandsTRA, cls).add_cmdline_args(argparser)
        super(PolyencoderAgent, cls).add_cmdline_args(argparser)
