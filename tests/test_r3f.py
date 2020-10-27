#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing r3f implementation.
"""

from parlai.nn.r3f import R3FMixin, R3F_NOISE_UNIFORM
import torch.nn as nn
import torch

import unittest

GENERATIVE = "generative"
DECODER_ONLY = "decoder_only"
# not doing this for encoder only yet, so not making a test case

DUMMY_MODEL_RETURN = (torch.Tensor(0), [])


class _DummyModuleWithEmbedding(nn.Module):
    def __init__(self, expect_noise_this, name, model_type):
        super(_DummyModuleWithEmbedding, self).__init__()
        self.expect_noise_this = expect_noise_this
        self.name = name
        self.model_type = model_type
        weight = torch.FloatTensor([[1]])
        self.norm_embeddings = nn.Embedding.from_pretrained(weight)
        self.had_first_pass = False

    def forward(self, *args, **kwargs):
        if self.had_first_pass:
            val = self.norm_embeddings(torch.LongTensor([0]))
            if self.expect_noise_this:
                assert (
                    val.item() != 1.0
                ), f"Did not noise embedding of '{self.name}' in '{self.model_type}' when expected"
            else:
                assert (
                    val.item() == 1.0
                ), f"Noised embedding '{self.name}' in '{self.model_type}' when not expected"
        self.had_first_pass = True


class _DummyGenerativeModel(nn.Module):
    """
    Dummy module with an encoder + decoder, and corresponding embeddings.
    """

    def __init__(self, opt):
        super(_DummyGenerativeModel, self).__init__()
        self.opt = opt
        self.encoder = _DummyModuleWithEmbedding(
            opt.get('r3f_encoder_noise'), 'encoder', GENERATIVE
        )
        self.decoder = _DummyModuleWithEmbedding(
            opt.get('r3f_decoder_noise'), 'decoder', GENERATIVE
        )

    def forward(self, *args, **kwargs):
        self.encoder()
        self.decoder()
        return DUMMY_MODEL_RETURN


class _DummyDecoderOnlyModelLikeGPT2(nn.Module):
    """
    Dummy class to test that R3F code is working properly on modles with only a decoder.

    R3F code finds decoder modules with a regex on the class name (currently looks for
    "GPT" as in "GPT2")
    """

    def __init__(self, opt):
        super(_DummyDecoderOnlyModelLikeGPT2, self).__init__()
        self.opt = opt
        self.decoder = _DummyModuleWithEmbedding(
            opt.get('r3f_decoder_noise'), 'decoder', DECODER_ONLY
        )

    def forward(self, *args, **kwargs):
        self.decoder()
        return DUMMY_MODEL_RETURN


######################### Following are wrapper abstractions to work with the fact that R3F is tightly coupled with how models in Parlai are set up.


class _DummyAgent(nn.Module):
    NULL_IDX = -1

    def __init__(self, module, opt):
        super(_DummyAgent, self).__init__()
        self.model = module
        self.opt = opt
        self._model_input = lambda x: [x]

    def compute_loss(self, *args, **kwargs):
        self.model()
        return (torch.Tensor(0), DUMMY_MODEL_RETURN)

    def forward(self, *args, **kwargs):
        return self.compute_loss(_DummyBatch())


class _DummyR3FAgent(R3FMixin, _DummyAgent):
    def __init__(self, module, opt):
        super(_DummyR3FAgent, self).__init__(module, opt)


class _DummyLabelVec:
    def ne(self, _):
        return torch.Tensor(0)


class _DummyBatch:
    label_vec = _DummyLabelVec()


####################################### End wrapper abstractions


class TestR3FEmbeddingHook(unittest.TestCase):
    """
    Test that R3F embedding hook combinations work appropriately.
    """

    def _get_base_opt(self):
        opt = {
            'r3f_lambda': 1000000000,  # High enough to overwhelm 1.0
            'r3f_eps': 1000,  # doesn't matter
            'r3f_noise_type': R3F_NOISE_UNIFORM,
        }
        return opt

    def _test_models(self, model_class, encoder_use, decoder_use):
        base_opt = self._get_base_opt()
        for a in encoder_use:
            for b in decoder_use:
                opt = {'r3f_encoder_noise': a, 'r3f_decoder_noise': b, **base_opt}
                agent = _DummyR3FAgent(model_class(opt), opt)
                agent()

    def test_generative_models(self):
        self._test_models(
            _DummyGenerativeModel, encoder_use=[True, False], decoder_use=[True, False]
        )

    def test_decoder_only_models(self):
        self._test_models(
            _DummyDecoderOnlyModelLikeGPT2,
            encoder_use=[False],
            decoder_use=[True, False],
        )


if __name__ == '__main__':
    unittest.main()
