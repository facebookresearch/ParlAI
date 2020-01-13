#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is an example how to extend torch ranker agent and use it for your own purpose.

In this example, we will just use a simple bag of words model.
"""
from parlai.core.torch_ranker_agent import TorchRankerAgent
import torch
from torch import nn


class ExampleBagOfWordsModel(nn.Module):
    """
    This constructs a simple bag of words model.

    It contains a encoder for encoding candidates and context.
    """

    def __init__(self, opt, dictionary):
        super().__init__()
        self.hidden_dim = opt.get('hidden_dim', 512)
        self.dict = dictionary
        self.encoder = nn.EmbeddingBag(len(self.dict), self.hidden_dim)

    def encode_text(self, text_vecs):
        """
        This function encodes a text_vec to a text encoding.
        """
        return self.encoder(text_vecs)

    def forward(self, batch, cand_vecs, cand_encs=None):
        bsz = cand_vecs.size(0)
        if cand_encs is None:
            if cand_vecs.dim() == 3:
                # if dim = 3, bsz *  num_candidates * seq_length
                # In this case, we are using inline candidates
                cand_vecs = cand_vecs.reshape(-1, cand_vecs.size(2))
            cand_encs = self.encode_text(cand_vecs)
            if cand_encs.size(0) != bsz:
                # Some cases, we could also use batch candidates,
                # they will be size of bsz * seq_length
                # so we don't have to use the extra treatment
                cand_encs = cand_encs.reshape(bsz, -1, self.hidden_dim)
        context_encodings = self.encode_text(batch.text_vec)
        if context_encodings.dim() != cand_encs.dim():
            return torch.sum(
                context_encodings.unsqueeze(1).expand_as(cand_encs) * cand_encs, 2
            )
        return context_encodings.mm(cand_encs.t())


class TraAgent(TorchRankerAgent):
    """
    Example subclass of TorchRankerAgent.

    This particular implementation is a simple bag-of-words model, which demonstrates
    the minimum implementation requirements to make a new ranking model.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add CLI args.
        """
        TorchRankerAgent.add_cmdline_args(argparser)
        arg_group = argparser.add_argument_group('ExampleBagOfWordsModel Arguments')
        arg_group.add_argument('--hidden-dim', type=int, default=512)

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        This function takes in a Batch object as well as a Tensor of candidate vectors.

        It must return a list of scores corresponding to the likelihood that the
        candidate vector at that index is the proper response. If `cand_encs` is not
        None (when we cache the encoding of the candidate vectors), you may use these
        instead of calling self.model on `cand_vecs`.
        """
        scores = self.model.forward(batch, cand_vecs, cand_encs)
        return scores

    def build_model(self):
        """
        This function is required to build the model and assign to the object
        `self.model`.
        """
        return ExampleBagOfWordsModel(self.opt, self.dict)
