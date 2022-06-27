# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper for ngram_repeat_block cuda extension.
"""
import torch
from torch import nn
from torch.autograd import Function


import os

current = os.getcwd()
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# import ngram_repeat_block_cuda

from torch.utils.cpp_extension import load

ngram_repeat_block_cuda = load(
    name='ngram_repeat_block_cuda',
    sources=[
        '../clib/cuda/ngram_repeat_block_cuda.cpp',
        '../clib/cuda/ngram_repeat_block_cuda_kernel.cu',
    ],
)
os.chdir(current)


class NGramRepeatBlockFunction(Function):
    """
    forward inputs to ngram_repeat_block cuda extension backward method not needed.
    """

    def forward(
        self,
        hypothesis,
        context,
        lprobs,
        bsz,
        step,
        beam_size,
        no_repeat_ngram_size,
        if_context_blocking,
    ):
        """
        Args:
        hypothesis(Tensor): (beam*bsz, current_sequence_length)
        context(Tensor): context for context-blocking
        lprobs(Tensor): likelihood probability(beam, vocab_size)
        bsz(int): batch size
        step(int): current step
        beam_size(int): beam size
        no_repeat_ngram_size(int): Ngram size
        if_context_blocking(bool): whether to use context-blocking
        """
        outputs = ngram_repeat_block_cuda.forward(
            hypothesis,
            context,
            lprobs,
            bsz,
            step,
            beam_size,
            no_repeat_ngram_size,
            if_context_blocking,
        )
        return outputs

    def backward(*args):
        raise NotImplementedError


class NGramRepeatBlock(nn.Module):
    """
    Wrapper class for calling ngram_repeat_block cuda extension.
    """

    def __init__(self):
        super(NGramRepeatBlock, self).__init__()

    def reset_parameters(self):
        pass

    def forward(
        self,
        hypothesis,
        context,
        lprobs,
        bsz,
        step,
        beam_size,
        no_repeat_ngram_size,
        if_context_blocking=False,
    ):
        """
        Args:
        hypothesis(Tensor): (beam*bsz, current_sequence_length)
        context(Tensor): context for context-blocking
        lprobs(Tensor): likelihood probability(beam, vocab_size)
        bsz(int): batch size
        step(int): current step
        beam_size(int): beam size
        no_repeat_ngram_size(int): Ngram size
        if_context_blocking(bool): whether to use context-blocking
        """
        # placeholder tensor to pass in to pass type check, won't be used
        if not if_context_blocking:
            context = torch.Tensor([0]).long()
        assert hypothesis.size(0) == bsz * beam_size
        assert lprobs.size(0) == bsz * beam_size
        hypothesis = hypothesis.contiguous()
        context = context.contiguous()
        lprobs = lprobs.contiguous()
        return NGramRepeatBlockFunction.apply(
            hypothesis,
            context,
            lprobs,
            bsz,
            step,
            beam_size,
            no_repeat_ngram_size,
            if_context_blocking,
        )
