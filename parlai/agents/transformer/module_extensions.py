#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implements extentions to NN code for transformers.

These include custom fine-tuning losses and the like that are training detail extentions
rather than new model architectures in themselves.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import AbstractContextManager

from parlai.core.params import ParlaiParser

R3F_NOISE_NORMAL = "normal"
R3F_NOISE_UNIFORM = "uniform"


class R3FNoiseContext(object):
    """
    Class that helps implement "Better Fine-Tuning by Reducing Representational
    Collapse", Aghajanyan et al. (2020). https://arxiv.org/ans/2008.03156.

    This class should be instantiated at a point that has access to command line args
    and the `compute_loss` function.
    """

    @staticmethod
    def add_cmdline_args(argparser: ParlaiParser):
        group = argparser.add_argument_group('R3F fine-tuning Args')
        group.add_argument(
            '--use-r3f',
            type=bool,
            default=False,
            help='should we use the R3f loss at all?',
        )
        group.add_argument('--r3f-eps', type=float, default=1e-5, help='noise eps')
        group.add_argument(
            '--r3f-lambda',
            type=float,
            default=1.0,
            help='lambda for combining logistic loss and noisy KL loss',
        )
        group.add_argument(
            '--r3f-noise-type',
            type=str,
            default=R3F_NOISE_UNIFORM,
            choices=[R3F_NOISE_NORMAL, R3F_NOISE_UNIFORM],
            help='type of noises for RXF methods',
        )
        group.add_argument(
            '--r3f-encoder-noise',
            type=bool,
            default=True,
            help='Add noise to encoder. At least one of `r3f-encoder-noise` and `r3f-coder-noise` must be set to True',
        )
        group.add_argument(
            '--r3f-decoder-noise',
            type=bool,
            default=False,
            help='Add noise to decoder. At least one of `r3f-encoder-noise` and `r3f-coder-noise` must be set to True',
        )

    def __init__(self, opts):
        self.use_r3f = opts.get('use_r3f')
        self.eps = opts.get('r3f_eps')
        self.noise_type = opts.get('r3f_noise_type')
        if self.noise_type == R3F_NOISE_NORMAL:
            self.noise_sampler = torch.distributions.normal.Normal(
                loc=0.0, scale=self.eps
            )
        elif self.noise_type == R3F_NOISE_UNIFORM:
            self.noise_sampler = torch.distributions.uniform.Uniform(
                low=-self.eps, high=self.eps
            )
        self.r3f_lambda = opts.get("r3f_lambda")
        self.noise_encoder = opts.get("r3f_encoder_noise")
        self.noise_decoder = opts.get("r3f_decoder_noise")

        if self.use_r3f and not self.noise_encoder and not self.noise_decoder:
            raise RuntimeError(
                "R3FContext: Noise must be added to at least one of the encoder or decoder."
            )
        self.had_first_pass = (
            False
        )  # necessary because nn.Embedding for encoder/decoder on some models are the same. Use this to only make any relevant deep copies once.

    def _calculate_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
        ) / noised_logits.size(0)

    def forward_pass_with_r3f(self, model, *args, **kwargs):
        """
        'model' argument here is the `nn.Module` normally responsible for the full
        forward pass.
        """
        if not self.use_r3f:
            raise RuntimeError(
                "Trying to use an R3F loss when this has not been allowed."
            )
        regular_output = model(*args, **kwargs)
        with R3FNoiseEmbeddingContextManager(self, model) as model:
            noised_output = model(*args, **kwargs)
        symm_kl = self._calculate_symm_kl(noised_output[0], regular_output[0])
        return regular_output, self.r3f_lambda * symm_kl


class R3FNoisedEmbedding(nn.Module):
    def __init__(self, noise_sampler):
        super(R3FNoisedEmbedding, self).__init__()
        self.noise_sampler = noise_sampler

    def __call__(self, module, input, output):
        noise = self.noise_sampler.sample(sample_shape=output.shape).to(output)
        return output + noise


class R3FNoiseEmbeddingContextManager(AbstractContextManager):
    """
    This class finds the first embedding(s) under the encoder/decoder, depending on what
    is selected, and adds R3F noise to their output.

    Note that this does not discrinate between the dictionary embedding lookup +
    position embedding classes. However, seeing as how addition is commutative, this
    should not be an issue.
    """

    def __init__(self, context, module):
        self.encoder_hook = None
        self.decoder_hook = None
        self.context = context
        for name, layer in module.named_modules():
            if context.noise_encoder and name == "encoder":
                self._find_embedding_and_set_hook(layer, "encoder")
            if context.noise_decoder and name == "decoder":
                self._find_embedding_and_set_hook(layer, "decoder")
        self.module = module
        self.context.had_first_pass = True

    def _find_embedding_and_set_hook(self, layer, name):
        try:
            parent, embedding_name = self._find_embedding(layer)
        except TypeError:
            raise RuntimeError(
                f"R3F noise: Could not find an embedding as a child of {name}. Does the model you are training use this type of layer? (Default R3F settings noises the encoder only, which some models - ex. GPT2 - do not use.)"
            )
        if not self.context.had_first_pass:
            # need to do a deep copy because models oftentimes share embedding object for encoder + decoder; breaks noise encoder/decoder only experiments
            setattr(
                parent, embedding_name, copy.deepcopy(getattr(parent, embedding_name))
            )
        embedding = getattr(parent, embedding_name)
        val = R3FNoisedEmbedding(self.context.noise_sampler)
        setattr(self, name + "_hook", embedding.register_forward_hook(val))

    def _find_embedding(self, parent):
        for child_name, child in parent.named_children():
            if child.__class__.__name__ == "Embedding":
                return parent, child_name
            recursion = self._find_embedding(child)
            if recursion is not None:
                return recursion
        return None

    def __enter__(self):
        return self.module

    def __exit__(self, type, value, traceback):
        if self.encoder_hook:
            self.encoder_hook.remove()
            self.encoder_hook = None
        if self.decoder_hook:
            self.decoder_hook.remove()
            self.decoder_hook = None
