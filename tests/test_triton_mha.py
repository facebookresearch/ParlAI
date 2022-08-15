import math
from typing import Dict, Tuple, Optional, Union

import triton
import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.opt import Opt
from parlai.core.params import default
from parlai.utils.torch import neginf
from parlai.agents.transformer.modules import _attention
import pytest
import parlai.utils.testing as testing_utils
import unittest


class Parlai_MHA(nn.Module):
    """
    Implements MultiHeadAttention; this is the core workhorse of the Transformer.

    See Vaswani (2017) for an extensive description.
    """

    def __init__(
        self, opt: Opt, n_heads: int = None, dim: int = None, dropout: float = 0
    ):
        super(Parlai_MHA, self).__init__()

        n_heads = default(n_heads, opt['n_heads'])
        dim = default(dim, opt['embedding_size'])

        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        static_kv: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        :param query: attention query
        :param key: attention key
        :param value: attention value
        :param mask: tensor in which True means that we are allowing attention and False
          means we are blocking it. Mask is:
          - [B, key_len] (encoder self-attn and decoder enc/dec attn)
          - [B, query_len, key_len] (decoder self-attn)
          - [B, 1, key_len] (decoder self-attn with incr_state caching)
        :param incr_state: dictionary with values representing the previous states of
          the key, value, and mask
        :param static_kv: True if the key and value are held constant during decoding
          (as in encoder/decoder attention)
        :return: (
          final attended tensor,
          new incremental state,
          key/value-multiplied tensor before softmax,
        )
        """

        batch_size, query_len, dim = query.size()

        assert (
            dim == self.dim
        ), 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                .contiguous()
                .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
            _, _key_len, dim = query.size()
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key

        assert key is not None  # let mypy know we sorted this
        _, _key_len, dim = key.size()

        q = prepare_head(query)

        # Prepend incremental states. For each of the key, value, and mask, see if
        # a previous incremental state exists, and if so, reshape it to match the shape
        # of the new state. Concatenate the previous and new states to match what the
        # full state would have been if we had not cached. (If we are using static_kv,
        # these three states are unchanging, so just re-use the cached states.)
        if incr_state is None:
            incr_state = {}
        if 'prev_key' in incr_state:
            prev_key = incr_state['prev_key'].view(
                batch_size * n_heads, -1, dim_per_head
            )
            if static_kv:
                k = prev_key
            else:
                k = torch.cat([prev_key, prepare_head(key)], dim=1)
        else:
            k = prepare_head(key)
        if 'prev_value' in incr_state:
            prev_value = incr_state['prev_value'].view(
                batch_size * n_heads, -1, dim_per_head
            )
            if static_kv:
                v = prev_value
            else:
                v = torch.cat([prev_value, prepare_head(value)], dim=1)
        else:
            v = prepare_head(value)
        if 'prev_mask' in incr_state:
            if static_kv:
                mask = incr_state['prev_mask']
            else:
                # Mask will be of size (B x query_len x key_len)
                # During incremental decoding the query will only represent the next token,
                # whereas the key/value will represent the entire sequence thus far.
                # In such a case, we only want to look at the last element of the mask in the query dimension.
                prev_mask = incr_state['prev_mask'][:, -query_len:, :]
                mask = torch.cat([prev_mask, mask], dim=2)
                # Prepend along the key_len dimension (analogous to incr_state['prev_key'])

        # Save new incremental states. We reshape to allow for reordering along batch
        # dimension.
        new_incr_state = {
            'prev_key': k.view(batch_size, n_heads, -1, dim_per_head),
            'prev_value': v.view(batch_size, n_heads, -1, dim_per_head),
            'prev_mask': mask,
        }

        full_key_len = k.size(1)
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, full_key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, full_key_len)
            .view(batch_size * n_heads, query_len, full_key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(
            dot_prod, dim=-1, dtype=torch.float  # type: ignore
        ).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        out = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, dim)
        )

        return out, new_incr_state, dot_prod


class Triton_MHA(nn.Module):
    """
    Implements MultiHeadAttention; this is the core workhorse of the Transformer.

    See Vaswani (2017) for an extensive description.
    """

    def __init__(
        self, opt: Opt, n_heads: int = None, dim: int = None, dropout: float = 0
    ):
        super(Triton_MHA, self).__init__()

        n_heads = default(n_heads, opt['n_heads'])
        dim = default(dim, opt['embedding_size'])

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        static_kv: bool = False,
        pad: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        :param query: attention query
        :param key: attention key
        :param value: attention value
        :param mask: tensor in which True means that we are allowing attention and False
          means we are blocking it. Mask is:
          - [B, key_len] (encoder self-attn and decoder enc/dec attn)
          - [B, query_len, key_len] (decoder self-attn)
          - [B, 1, key_len] (decoder self-attn with incr_state caching)
        :param incr_state: dictionary with values representing the previous states of
          the key, value, and mask
        :param static_kv: True if the key and value are held constant during decoding
          (as in encoder/decoder attention)
        :return: (
          final attended tensor,
          new incremental state,
          key/value-multiplied tensor before softmax,
        )
        """

        def pad(tensor):
            _, old, _ = tensor.size()
            return F.pad(
                tensor,
                pad=(0, 0, 0, 128 * math.ceil(old / 128) - old),
                mode='constant',
                value=0,
            )

        def add_padding(query, key, value, mask):
            _, old_query_len, _ = query.size()
            query = pad(query)
            if key != None:
                key = pad(key)
            if value != None:
                value = pad(value)
            mask = F.pad(
                mask,
                pad=(0, 128 * math.ceil(old_query_len / 128) - old_query_len),
                mode='constant',
                value=0,
            )
            return query, key, value, mask, old_query_len

        if pad:
            query, key, value, mask, old_query_len = add_padding(
                query, key, value, mask
            )

        batch_size, query_len, dim = query.size()

        assert (
            dim == self.dim
        ), 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = 1.0 / math.sqrt(dim_per_head)

        if key is None and value is None:
            # self attention
            key = value = query
            _, _key_len, dim = query.size()
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key

        def prepare_head(tensor):
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(bsz, seq_len, n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous()
            return tensor

        q = prepare_head(query)
        k = prepare_head(key)
        v = prepare_head(value)

        full_key_len = k.size(2)

        mask = (
            (mask == 0)
            .view(batch_size, 1, -1, full_key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, full_key_len)
            .clone()
        )

        dropout = None
        if self.dropout != 0:
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.dropout)
            dropout = binomial.sample(
                (batch_size, n_heads, query_len, query_len)
            ).cuda()

        attention = _attention.apply
        out = attention(q, k, v, mask, scale, dim_per_head, dropout, self.dropout)

        if pad:
            query_len = old_query_len

        out = out[:, :, :query_len, :]

        out = (
            out.type_as(query)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, dim)
        )

        return out, {}


class TestTritonAttentionKernel(unittest.TestCase):
    @testing_utils.skipUnlessGPU
    def test_self_attention(self):
        opt = Opt()
        Z, H, N_CTX, D_HEAD = 64, 16, 128, 32
        DIM = H * D_HEAD
        dtype = torch.float16
        torch.manual_seed(42)
        query = (
            torch.empty((Z, N_CTX, DIM), dtype=dtype, device="cuda")
            .normal_(mean=0, std=0.5)
            .requires_grad_()
        )
        mask = torch.ones(Z, N_CTX, dtype=torch.bool, device="cuda")
        mask[:, -5:] = False
        opt['n_heads'] = H
        opt['embedding_size'] = DIM
        parlai_mha = Parlai_MHA(opt)
        triton_mha = Triton_MHA(opt)

        # parlai
        out, _, _ = parlai_mha(query, mask=mask, pad=False)

        # triton
        tri_out, _ = triton_mha(query, mask=mask, pad=False)

        assert not tri_out.isnan().any(), "Triton output contains Nan."
        assert (
            tri_out - out
        ).abs().max() < 0.01, "Triton and Parlai output differs for more than 0.01."

    def test_enc_dec_attention(self):
        # ENCODER-DECODER: mask = torch.Size([64, 24, 24])
        opt = Opt()
        Z, H, N_CTX, D_HEAD = 64, 16, 24, 32
        DIM = H * D_HEAD
        dtype = torch.float16
        torch.manual_seed(42)
        query = (
            torch.empty((Z, N_CTX, DIM), dtype=dtype, device="cuda")
            .normal_(mean=0, std=0.5)
            .requires_grad_()
        )
        key = (
            torch.empty((Z, N_CTX, DIM), dtype=dtype, device="cuda")
            .normal_(mean=0, std=0.5)
            .requires_grad_()
        )
        value = (
            torch.empty((Z, N_CTX, DIM), dtype=dtype, device="cuda")
            .normal_(mean=0, std=0.5)
            .requires_grad_()
        )
        # mask = torch.randint(0, 2, (Z, N_CTX, N_CTX), dtype=torch.bool, device="cuda")
        mask = torch.ones(Z, N_CTX, dtype=torch.bool, device="cuda")
        mask[:, -5:] = False
        opt['n_heads'] = H
        opt['embedding_size'] = DIM
        parlai_mha = Parlai_MHA(opt)
        triton_mha = Triton_MHA(opt)

        # parlai
        out, _, _ = parlai_mha(query, key, value, mask=mask, pad=False)

        # triton
        tri_out, _ = triton_mha(query, key, value, mask=mask, pad=True)

        assert not tri_out.isnan().any(), "Triton output contains Nan."
        assert (
            tri_out - out
        ).abs().max() < 0.01, "Triton and Parlai output differs for more than 0.01."

    def test_self_attention_dropout(self):
        opt = Opt()
        Z, H, N_CTX, D_HEAD = 64, 16, 128, 32
        DIM = H * D_HEAD
        dtype = torch.float16
        torch.manual_seed(42)
        query = (
            torch.empty((Z, N_CTX, DIM), dtype=dtype, device="cuda")
            .normal_(mean=0, std=0.5)
            .requires_grad_()
        )
        mask = torch.ones(Z, N_CTX, dtype=torch.bool, device="cuda")
        mask[:, -5:] = False
        opt['n_heads'] = H
        opt['embedding_size'] = DIM
        parlai_mha = Parlai_MHA(opt, dropout=0.5)
        triton_mha = Triton_MHA(opt, dropout=0.5)

        # parlai
        out, _, _ = parlai_mha(query, mask=mask, pad=False)

        # triton
        tri_out, _ = triton_mha(query, mask=mask, pad=False)

        assert not tri_out.isnan().any(), "Triton output contains Nan."
        assert (
            tri_out - out
        ).abs().max() < 0.01, "Triton and Parlai output differs for more than 0.01."


# max head we can do right now is 42
Z, N_HEADS, N_CTX, D_HEAD = 3, 36, 4096, 64
# vary seq length for fixed head and batch=4
configs = [
    triton.testing.Benchmark(
        x_names=['N_CTX'],
        x_vals=[2**i for i in range(9, 13)],
        line_arg='provider',
        line_vals=['triton', 'parlai'],
        line_names=['Triton', 'ParlAI'],
        styles=[('red', '-'), ('blue', '-')],
        ylabel='ms',
        plot_name=f'fused-attention-batch{Z}-head{N_HEADS}-d{D_HEAD}-{mode}',
        args={
            'H': N_HEADS,
            'Z': Z,
            'D_HEAD': D_HEAD,
            'dtype': torch.float16,
            'mode': mode,
        },
    )
    for mode in ['fwd']
]


@triton.testing.perf_report(configs)
def bench_flash_attention(
    Z, H, N_CTX, D_HEAD, mode, provider, dtype=torch.float16, device="cuda"
):
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 10
    DIM = H * D_HEAD
    q = (
        torch.empty((Z, N_CTX, DIM), dtype=dtype, device="cuda")
        .normal_(mean=0, std=0.5)
        .requires_grad_()
    )
    m = torch.ones(Z, N_CTX, dtype=torch.bool, device="cuda")
    m[:, -5:] = False

    opt = Opt()
    opt['n_heads'] = H
    opt['embedding_size'] = DIM

    if provider == "triton":
        triton_mha = Triton_MHA(opt)
        fn = lambda: triton_mha.forward(q, mask=m, padding=True)
        ms = triton.testing.do_bench(fn, percentiles=None, warmup=warmup, rep=rep)
        return ms
    if provider == "parlai":
        parlai_mha = Parlai_MHA(opt)
        fn = lambda: parlai_mha.forward(q, mask=m)
        ms = triton.testing.do_bench(fn, percentiles=None, warmup=warmup, rep=rep)
        return ms


# only works on A100 at the moment
bench_flash_attention.run(save_path='.', print_data=True)
