#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implements extentions to NN code for transformers.

These include custom fine-tuning losses and the like that are training detail extentions
applilcable to transformers rather than new model architectuures in themselves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.params import ParlaiParser

R3F_NOISE_NORMAL = "normal"
R3F_NOISE_UNIFORM = "uniform"


class R3FNoiseContext(object):
    @staticmethod
    def add_cmdline_args(argparser: ParlaiParser):
        group = argparser.add_argument_group('R3F fine-tuning Args')
        group.add_argument(
            '--use-r3f',
            type=bool,
            default=False,  # TODO: Change this before landing. Else everyone is getting this loss (with BART)
            help='should we use the R3f loss at all?',
        )
        group.add_argument('--eps', type=float, default=1e-5, help='noise eps')
        group.add_argument(
            '--r3f-lambda',
            type=float,
            default=1.0,
            help='lambda for combining logistic loss and noisy KL loss',
        )
        group.add_argument(
            '--noise-type',
            type=str,
            default=R3F_NOISE_UNIFORM,
            choices=[R3F_NOISE_NORMAL, R3F_NOISE_UNIFORM],
            help='type of noises for RXF methods',
        )

    def __init__(self, opts):
        self.use_r3f = opts.get('use_r3f')
        self.eps = opts.get('eps')
        self.noise_type = opts.get('noise_type')
        if self.noise_type == R3F_NOISE_NORMAL:
            self.noise_sampler = torch.distributions.normal.Normal(
                loc=0.0, scale=self.eps
            )
        elif self.noise_type == R3F_NOISE_UNIFORM:
            self.noise_sampler = torch.distributions.uniform.Uniform(
                low=-self.eps, high=self.eps
            )
        self.r3f_lambda = opts.get("r3f_lambda")

        self._in_noise_pass = False  # state flag that toggles what's happening
        self._embeddinig_noised_this_pass = False  # debug value

    def calculate_symm_kl(self, noised_logits, input_logits):
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

    def get_embedding_extension(self):
        if self.use_r3f:
            return R3FNoiseEmbeddingExtension(self)
        return None

    def get_loss_extension(self):
        if self.use_r3f:
            return R3FNoiseLossExtension(self)
        return None

    def start_noise_pass(self):
        if self._in_noise_pass:
            raise RuntimeError(
                "R3FContext: We should never be in starting a noised pass while already in one."
            )
        if self._embeddinig_noised_this_pass:
            raise RuntimeError(
                "R3FContext: We are just starting a noised pass. The embedding should not have been noised yet."
            )
        self._in_noise_pass = True
        self._embeddinig_noised_this_pass = False

    def mark_embedding_noised(self):
        # NOTE: Might need to change this if we add noise to both encoder + decoder rather than just encoder
        if self._embeddinig_noised_this_pass:
            raise RuntimeError(
                "R3FContext: An input embedding should not be noised if one already has been."
            )
        self._embeddinig_noised_this_pass = True

    def end_noise_pass(self):
        if not self._in_noise_pass:
            raise RuntimeError(
                "R3FContext: Ending a noised pass when one has not been started is nonsensical."
            )
        if not self._embeddinig_noised_this_pass:
            raise RuntimeError(
                "R3FContext: We are trying to end a noised pass when we have not actuallly done any noising."
            )
        self._in_noise_pass = False
        self._embeddinig_noised_this_pass = False

    def in_noise_pass(self):
        return self._in_noise_pass


class R3FNoiseEmbeddingExtension(object):
    """
    Provides a way to hot-swap out the input-to-embedding function to one that also adds
    noise.

    Note that this needs to be called within a class that calls the "forward" which adds
    said noise.
    """

    def __init__(self, context: R3FNoiseContext):
        print("R3fEmbedding Init")
        self.context = context

    def _noised_embedding_func(self, base_embedding):
        noise_sampler = self.noise_sampler
        # NOTE: Not sure if this is the best way to define the callable. Maybe use a lambda here? Maybe use some other python-function-magic that I don't know about? Suggestions welcome.

        def _result(input):
            nonlocal base_embedding
            nonlocal noise_sampler
            tensor = base_embedding(input)
            noise = noise_sampler.sample(sample_shape=tensor.shape).to(tensor)
            return tensor + noise

        return _result

    def forward(self, base, *args, **kwargs):
        """
        Hot swaps out the embedding calculation in `base_module` to be one that returns
        noiseed embeddings, runs the forward as 'normal', then puts the original
        embedding back.
        """
        if not self.context.use_r3f:
            raise RuntimeError(
                "Trying to add embedding noise for use in an R3f loss when using an R3F loss has not been allowed."
            )
        if self.context.in_noised_pass():
            original_embedding = base.embeddings
            base.embeddings = self._noised_embedding_func(original_embedding)
            result = base.forward(*args, **kwargs)
            base.embeddings = original_embedding
            self.context.mark_embedding_noised()
            return result
        else:
            return base.forward(*args, **kwargs)


class R3FNoiseLossExtension(object):
    """
    This class should be colocated with whatever normally calls `compute_loss` with
    access to a model.

    In the generator case, this will normally be a TorchGeneratorAgent or a subclass
    thereof.
    """

    def __init__(self, context: R3FNoiseContext):
        self.context = context

    def forward_pass_with_r3f(self, model, *args, **kwargs):
        if not self.context.use_r3f:
            raise RuntimeError(
                "Trying to use an R3F loss when this has not been allowed."
            )
        regular_output = model(*args, **kwargs)
        self.context.start_noise_pass()
        noised_output = model(*args, **kwargs)
        self.context.end_noise_pass()

        symm_kl = self.context.calculate_symm_kl(noised_output[0], regular_output[0])
        return regular_model_output, self.context.r3f_lambda * symm_kl
