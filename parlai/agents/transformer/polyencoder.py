# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure -m transformer/generator works as expected
from .modules import TransformerEncoder
from .modules import get_n_positions_from_options
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
        agent = argparser.add_argument_group('Polyencoder Arguments')
        agent.add_argument(
            '--polyencoder-type',
            type=str,
            default='codes',
            choices=['codes', 'n_first'],
            help='Type of polyencoder, either we compute'
            'vectors using codes + attention, or we '
            'simply take the first N vectors.',
        )
        agent.add_argument(
            '--poly-n-codes',
            type=int,
            default=64,
            help='number of vectors used to represent the context'
            'in the case of n_first, those are the number'
            'of vectors that are considered.',
        )
        agent.add_argument(
            '--poly-attention-type',
            type=str,
            default='basic',
            choices=['basic', 'sqrt', 'multihead'],
            help='Type of the top aggregation layer of the poly-'
            'encoder (where the candidate representation is'
            'the key)',
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
            from parlai.core.distributed_utils import is_distributed

            if is_distributed():
                raise ValueError('Cannot combine --data-parallel and distributed mode')
            self.model = torch.nn.DataParallel(self.model)

    def build_model(self, states=None):
        self.model = PolyEncoderModule(self.opt, self.dict, self.NULL_IDX)
        return self.model

    def vectorize(self, *args, **kwargs):
        """ Add the start and end token to the labels.
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

    def vectorize_fixed_candidates(self, *args, **kwargs):
        """ Add the start and end token when computing the candidate encodings
            in interactive mode.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        return super().vectorize_fixed_candidates(*args, **kwargs)

    def _make_candidate_encs(self, vecs, path):
        """ (used in interactive mode only) The polyencoder module expects
            cand vecs to be 3D while torch_ranker_agent expects it to be 2D.
            This requires a little adjustment
        """
        rep = super()._make_candidate_encs(vecs, path)
        return rep.transpose(0, 1).contiguous()

    def encode_candidates(self, padded_cands):
        padded_cands = padded_cands.unsqueeze(1)
        _, _, cand_rep = self.model(cand_tokens=padded_cands)
        return cand_rep

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        bsz = batch.text_vec.size(0)
        ctxt_rep, ctxt_rep_mask, _ = self.model(ctxt_tokens=batch.text_vec)

        if cand_encs is not None:
            if bsz == 1:
                cand_rep = cand_encs
            else:
                cand_rep = cand_encs.expand(bsz, cand_encs.size(1), -1)
        elif len(cand_vecs.shape) == 3:
            _, _, cand_rep = self.model(cand_tokens=cand_vecs)
        elif len(cand_vecs.shape) == 2:
            _, _, cand_rep = self.model(cand_tokens=cand_vecs.unsqueeze(1))
            cand_rep = cand_rep.expand(bsz, bsz, -1).transpose(0, 1).contiguous()
        scores = self.model(
            ctxt_rep=ctxt_rep, ctxt_rep_mask=ctxt_rep_mask, cand_rep=cand_rep
        )
        return scores


class PolyEncoderModule(torch.nn.Module):
    """ See https://arxiv.org/abs/1905.01969
    """

    def __init__(self, opt, dict, null_idx):
        super(PolyEncoderModule, self).__init__()
        self.null_idx = null_idx
        self.encoder_ctxt = self.get_encoder(opt, dict, null_idx, 'none')
        self.encoder_cand = self.get_encoder(opt, dict, null_idx, opt['reduction_type'])

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
            codes = torch.empty(self.n_codes, embed_dim)
            codes = torch.nn.init.uniform_(codes)
            self.codes = torch.nn.Parameter(codes)

            # The attention for the codes.
            if self.codes_attention_type == 'multihead':
                self.code_attention = MultiHeadAttention(
                    self.codes_attention_num_heads, embed_dim, opt['dropout']
                )
            elif self.codes_attention_type == 'sqrt':
                self.code_attention = BasicAttention(
                    dim=2, attn='sqrt', get_weights=False
                )
            elif self.codes_attention_type == 'basic':
                self.code_attention = BasicAttention(
                    dim=2, attn='basic', get_weights=False
                )

        # The final attention (the one that takes the candidate as key)
        if self.attention_type == 'multihead':
            self.attention = MultiHeadAttention(
                self.attention_num_heads, opt['embedding_size'], opt['dropout']
            )
        else:
            self.attention = BasicAttention(
                dim=2, attn=self.attention_type, get_weights=False
            )

    def get_encoder(self, opt, dict, null_idx, reduction_type):
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

    def attend(self, attention_layer, queries, keys, mask):
        """ Unify the API of MultiHeadAttention and
            BasicAttention that are slighlty different
        """
        if isinstance(attention_layer, BasicAttention):
            return attention_layer(queries, keys, mask)
        elif isinstance(attention_layer, MultiHeadAttention):
            return attention_layer(queries, keys, None, mask)
        else:
            raise Exception('Unrecognized type of attention')

    def encode(self, ctxt_tokens, cand_tokens):
        """
            :param ctxt_tokens:
                2D long tensor, batchsize x sent_len
            :param cand_tokens:
                3D long tensor, batchsize x num_cands x sent_len
                Note this will actually view it as a 2D tensor
            :returns: (ctxt_rep, ctxt_mask, cand_rep)
                - ctxt_rep 3D float tensor, batchsize x n_codes x dim
                - ctxt_mask byte:  batchsize x n_codes (all 1 in case
                    of polyencoder with code. Which are the vectors to use
                    in the ctxt_rep)
                - cand_rep (3D float tensor) batchsize x num_cands x dim
        """
        cand_embed = None
        ctxt_rep = None
        ctxt_rep_mask = None

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
            ctxt_out, ctxt_mask = self.encoder_ctxt(ctxt_tokens)
            dim = ctxt_out.size(2)

            if self.type == 'codes':
                ctxt_rep = self.attend(
                    self.code_attention,
                    self.codes.repeat(bsz, 1, 1),
                    ctxt_out,
                    ctxt_mask,
                )
                ctxt_rep_mask = ctxt_rep.new_ones(bsz, self.n_codes).byte()

            elif self.type == 'n_first':
                # Expand the output if it is not long enough
                if ctxt_out.size(1) < self.n_codes:
                    difference = self.n_codes - ctxt_out.size(1)
                    extra_rep = ctxt_out.new_zeros(bsz, difference, dim)
                    ctxt_rep = torch.cat([ctxt_out, extra_rep], dim=1)
                    extra_mask = ctxt_mask.new_zeros(bsz, difference)
                    ctxt_rep_mask = torch.cat([ctxt_mask, extra_mask], dim=1)
                else:
                    ctxt_rep = ctxt_out[:, 0 : self.n_codes, :]
                    ctxt_rep_mask = ctxt_mask[:, 0 : self.n_codes]

        return ctxt_rep, ctxt_rep_mask, cand_embed

    def score(self, ctxt_rep, ctxt_rep_mask, cand_embed):
        """
            Scores the candidates
            :param ctxt_rep: 3D float tensor, bsz x ctxt_len x dim
            :param ctxt_rep_mask: 2D byte tensor, bsz x ctxt_len, in case
                there are some elements of the ctxt that we should not take into
                account.
            :param cand_embed: 3D float tensor, bsz x num_cands x dim

            :returns: scores, 2D float tensor: bsz x num_cands
        """

        # reduces the context representation to a 3D tensor bsz x num_cands x dim
        ctxt_final_rep = self.attend(
            self.attention, cand_embed, ctxt_rep, ctxt_rep_mask
        )
        scores = torch.sum(ctxt_final_rep * cand_embed, 2)
        return scores

    def forward(
        self,
        ctxt_tokens=None,
        cand_tokens=None,
        ctxt_rep=None,
        ctxt_rep_mask=None,
        cand_rep=None,
    ):
        """ Due to a limitation of parlai, we have to have one single model
            in the agent. And because we want to be able to use data-parallel,
            we need to have one single forward() method.
            Therefore the operation_type can be either 'encode' or 'score'.
        """
        if ctxt_tokens is not None or cand_tokens is not None:
            return self.encode(ctxt_tokens, cand_tokens)
        elif (
            ctxt_rep is not None and ctxt_rep_mask is not None and cand_rep is not None
        ):
            return self.score(ctxt_rep, ctxt_rep_mask, cand_rep)
        raise Exception('Unsupported operation')
