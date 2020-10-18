#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implements extentions to NN code for transformers.

These include custom fine-tuning losses and the like that are training detail extentions
rather than new model architectures in themselves.
"""

from contextlib import AbstractContextManager
import copy
import re
import torch
import torch.nn.functional as F

from parlai.core.params import ParlaiParser

R3F_NOISE_NORMAL = "normal"
R3F_NOISE_UNIFORM = "uniform"


class R3FMixin(object):
    """
    Class that helps implement "Better Fine-Tuning by Reducing Representational
    Collapse", Aghajanyan et al.

    (2020). https://arxiv.org/ans/2008.03156.
    """

    @staticmethod
    def add_cmdline_args(argparser: ParlaiParser):
        group = argparser.add_argument_group('R3F fine-tuning Args')
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
        return argparser

    def set_r3f_settings_from_opts(self, opts):
        self.r3f_eps = opts.get('r3f_eps')
        self.r3f_noise_type = opts.get('r3f_noise_type')
        if self.r3f_noise_type == R3F_NOISE_NORMAL:
            self.r3f_noise_sampler = torch.distributions.normal.Normal(
                loc=0.0, scale=self.r3f_eps
            )
        elif self.r3f_noise_type == R3F_NOISE_UNIFORM:
            self.r3f_noise_sampler = torch.distributions.uniform.Uniform(
                low=-self.r3f_eps, high=self.r3f_eps
            )
        self.r3f_lambda = opts.get("r3f_lambda")

        # NOTE: Following is temporary until it is clear whether or not to noise both or just one of encoder/decoder
        self.noise_encoder = opts.get("r3f_encoder_noise")
        self.noise_decoder = opts.get("r3f_decoder_noise")

        # Find embedding values and store locally so we don't have to find them each turn
        self.r3f_embeddings = {}
        self._deep_copy_encoder_embedding(self.model)  # temporary
        self._find_embeddings(self.model)

    def compute_loss(self, batch, return_output=False):
        if not hasattr(self, 'r3f_lambda'):
            self.set_r3f_settings_from_opts(self.opt)
        loss, standard_output = super().compute_loss(batch, True)
        with R3FNoiseEmbeddingContextManager(self, self.model) as r3f:
            noised_scores, _, *_ = self.model(
                *self._model_input(batch), ys=batch.label_vec
            )
            standard_scores, _, *_ = standard_output
            r3f_loss = r3f._calculate_symm_kl(noised_scores, standard_scores)
            # get average loss per token correctly
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum(dim=-1)
            r3f_loss /= target_tokens.sum()
            # add in loss
            loss += self.r3f_lambda * r3f_loss
        if return_output:
            return (loss, standard_output)
        else:
            return loss

    def _calculate_symm_kl(self, noised_logits, input_logits):
        return F.kl_div(
            F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
            F.softmax(input_logits, dim=-1, dtype=torch.float32),
            None,
            None,
            "sum",
        ) + F.kl_div(
            F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
            F.softmax(noised_logits, dim=-1, dtype=torch.float32),
            None,
            None,
            "sum",
        )

    def _find_embeddings(self, module):
        for name, layer in module.named_modules():
            if self.noise_encoder and (re.search(f"encoder.*norm_embeddings$", name)):
                self.r3f_embeddings["encoder"] = layer
            if self.noise_decoder and (
                re.search(f"decoder.*norm_embeddings$", name)
                or re.search("decoder.*wte$", name)
            ):
                self.r3f_embeddings["decoder"] = layer
        if (self.noise_encoder is True and "encoder" not in self.r3f_embeddings) or (
            self.noise_decoder is True and "decoder" not in self.r3f_embeddings
        ):
            raise RuntimeError(
                """
                R3F: Embedding to add noise to not found in model. (Does the part, encoder/decoder, exist in the model you are running? Is the name for the embedding accounted for in `_find_embeddings()`?)
                """
            )

    def _deep_copy_encoder_embedding(self, parent, found_encoder=False):
        """
        TEMPORARY UNTIL BEST NOISING MODE (encoder only, decoder only, both) determined.

        Needed cause BART has the same embedding in both the encoder/decoder; use this
        to separate them.
        """
        for name, layer in parent.named_children():
            if name == "encoder":
                self._deep_copy_encoder_embedding(layer, True)
            elif found_encoder and name == "norm_embeddings":
                parent.norm_embeddings = copy.deepcopy(parent.norm_embeddings)
            elif found_encoder and name == "wte":
                parent.wte = copy.deepcopy(parent.wte)
            else:
                self._deep_copy_encoder_embedding(layer, found_encoder)


class R3FNoiseEmbeddingContextManager(AbstractContextManager):
    """
    This class finds the input embeddings map under the encoder/decoder, depending on
    what is selected, and adds R3F noise to their output.
    """

    def __init__(self, context, module):
        self.encoder_hook = None
        self.decoder_hook = None
        self.context = context
        self.hooks = {}

        if self.context.noise_encoder:
            self.hooks["encoder"] = self.context.r3f_embeddings[
                "encoder"
            ].register_forward_hook(self._hook_implementation)
        if self.context.noise_decoder:
            self.hooks["decoder"] = self.context.r3f_embeddings[
                "decoder"
            ].register_forward_hook(self._hook_implementation)

    def __enter__(self):
        return self.context

    def __exit__(self, type, value, traceback):
        for key in self.hooks.keys():
            if self.hooks[key]:
                self.hooks[key].remove()
            self.hooks[key] = None

    def _hook_implementation(self, module, input, output):
        noise = self.context.r3f_noise_sampler.sample(sample_shape=output.shape).to(
            output
        )
        return output + noise
