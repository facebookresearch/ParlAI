import math
from typing import Dict, Tuple, Optional, Union

import triton
import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.opt import Opt
from parlai.core.params import default
from parlai.utils.torch import neginf
from triton_att import _attention


class MultiHeadAttention(nn.Module):
    """
    Implements MultiHeadAttention; this is the core workhorse of the Transformer.

    See Vaswani (2017) for an extensive description.
    """

    def __init__(self, n_heads: int = None):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads

    def forward(  # type: ignore
        # TODO: remove type ignore with pytorch 1.5:
        # https://github.com/pytorch/pytorch/pull/31057
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        dropout_mask: torch.Tensor = None,
        p: float = 0,
        static_kv: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size, query_len, dim = query.size()
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                .contiguous()
                .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        _, _key_len, dim = key.size()

        q = prepare_head(query)
        k = prepare_head(key)
        v = prepare_head(value)

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
        attn_weights *= dropout_mask * (1.0 / (1 - p))

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, dim)
        )

        return attentioned


def prepare_head(tensor):
    Z, H, N_CTX, D_HEAD = tensor.size()
    return tensor.transpose(1, 2).contiguous().view(Z, N_CTX, H * D_HEAD)


def slow_test():
    Z, H, N_CTX, D_HEAD = 3, 2, 2048, 64
    dtype = torch.float16
    torch.manual_seed(42)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0, std=0.5)
        .requires_grad_()
    )
    # a bitmask with approximately 80% zero's
    m = torch.ones(Z, N_CTX).cuda()
    m[:, -5:] = 0
    mask_reshape = (
        m.view(Z, 1, -1, N_CTX).repeat(1, H, 1, 1).expand(Z, H, N_CTX, N_CTX)
    ).clone()  # clone() to make stride correct
    sm_scale = 1 / math.sqrt(D_HEAD)
    p = 0.3
    binomial = torch.distributions.binomial.Binomial(probs=1 - p)
    dropout_mask = binomial.sample((Z * H, N_CTX, N_CTX)).cuda()
    dropout_mask_reshape = dropout_mask.view(Z, H, N_CTX, N_CTX)

    # parlai
    a = MultiHeadAttention(n_heads=H)
    # out shape -> [Z, N_CTX, H * D_HEAD]
    out = a.forward(
        prepare_head(q), prepare_head(k), prepare_head(v), m, dropout_mask, p
    )
    out = out.view(Z, N_CTX, H, D_HEAD).transpose(1, 2).contiguous()

    # triton
    attention = _attention.apply
    tri_out = attention(q, k, v, mask_reshape, sm_scale, dropout_mask_reshape, p)

    # pytorch
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)

    print("max diff between parlai and triton: ", (tri_out - out).max())


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
    q = torch.randn(
        (Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
    )
    k = torch.randn(
        (Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
    )
    v = torch.randn(
        (Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True
    )
    m = torch.cuda.FloatTensor(Z, N_CTX).uniform_() > 0.8
    mask_reshape = (
        m.view(Z, 1, -1, N_CTX).repeat(1, H, 1, 1).expand(Z, H, N_CTX, N_CTX)
    ).clone()  # clone() to make stride correct
    p = 0.2
    binomial = torch.distributions.binomial.Binomial(probs=1 - p)
    dropout_mask = binomial.sample((Z * H, N_CTX, N_CTX)).cuda()
    dropout_mask_reshape = dropout_mask.view(Z, H, N_CTX, N_CTX)

    if provider == "triton":
        sm_scale = 1 / math.sqrt(D_HEAD)
        attention = _attention.apply
        fn = lambda: attention(q, k, v, mask_reshape, sm_scale, dropout_mask_reshape, p)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, percentiles=None, warmup=warmup, rep=rep)
        return ms
    if provider == "parlai":
        a = MultiHeadAttention(n_heads=H)
        fn = lambda: a.forward(
            prepare_head(q), prepare_head(k), prepare_head(v), m, dropout_mask, p
        )
        ms = triton.testing.do_bench(fn, percentiles=None, warmup=warmup, rep=rep)
        return ms


# only works on A100 at the moment
slow_test()
bench_flash_attention.run(save_path='.', print_data=True)
