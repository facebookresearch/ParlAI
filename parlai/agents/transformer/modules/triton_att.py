#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Triton fused attention code adapted from:
# https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    MASK,
    DROPOUT,
    sm_scale,
    dropout_p,
    add_dropout,
    TMP,
    L,
    M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_dz,
    stride_dh,
    stride_d3,
    stride_d4,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = (
        off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    )
    off_k = (
        off_hz * stride_qh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    )
    off_v = (
        off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    )
    off_mask = (
        off_hz * stride_dh + offs_m[:, None] * stride_d3 + offs_n[None, :] * stride_d4
    )
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    mask_ptrs = MASK + off_mask
    dropout_ptrs = DROPOUT + off_mask  # mask and dropout shares the same blocks
    # initialize pointer to m and l
    t_ptrs = TMP + off_hz * N_CTX + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    for start_n in range(0, N_CTX, BLOCK_N):
        # TODO: the following line currently gives a compiler error, not sure why
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k and mask ----
        k = tl.load(k_ptrs + start_n * stride_kn)
        mask = tl.load(mask_ptrs + start_n)
        drop = tl.load(dropout_ptrs + start_n)
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, trans_b=True)
        qk *= sm_scale
        qk += tl.where(mask == 1, -1e20, 0)
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        if add_dropout:
            p *= tl.where(drop != 0, drop / (1 - dropout_p), 0.0)
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(v_ptrs + start_n * stride_vk)
        p = p.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = (
        off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, sm_scale, dmodel_size, dropout=None, dropout_p=0.0):
        BLOCK = 128
        # shape constraints
        Lq, Lk = q.shape[-1], k.shape[-1]
        assert Lq == Lk
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
        tmp = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        L = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        m = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        add_dropout = True

        if dropout == None:
            dropout = torch.zeros(mask.shape, device=q.device, dtype=torch.float32)
            add_dropout = False

        _fwd_kernel[grid](
            q,
            k,
            v,
            mask,
            dropout,
            sm_scale,
            dropout_p,
            add_dropout,
            tmp,
            L,
            m,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            mask.stride(0),
            mask.stride(1),
            mask.stride(2),
            mask.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=dmodel_size,
            num_warps=4,
            num_stages=1,
        )
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.BLOCK = BLOCK
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = dmodel_size
        return o
