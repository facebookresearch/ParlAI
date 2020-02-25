#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Subclass of TorchAgent used for handling image features.
"""

from abc import abstractmethod
from typing import List

import torch

from parlai.core.message import Message
from parlai.core.torch_agent import Batch, TorchAgent


class TorchImageAgent(TorchAgent):
    """
    Subclass of TorchAgent that allows for encoding image features.

    Provides flags and utility methods.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        super(TorchImageAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Image args')
        agent.add_argument(
            '--image-features-dim',
            type=int,
            default=2048,
            help='Dimensionality of image features',
        )
        agent.add_argument(
            '--image-encoder-num-layers',
            type=int,
            default=1,
            recommended=1,
            help='Number of linear layers to encode image features with',
        )
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.image_features_dim = opt['image_features_dim']
        self.image_encoder_num_layers = opt['image_encoder_num_layers']

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        """
        Override to handle image features.
        """
        batch = super().batchify(obs_batch, sort)
        batch = self.batchify_image_features(batch)
        return batch

    @abstractmethod
    def batchify_image_features(self, batch: Batch) -> Batch:
        """
        Put this batch of images into the correct format for this agent.

        self._process_image_features() will likely be useful for this.
        """
        raise NotImplementedError(
            'Subclasses must implement method for batching images!'
        )

    def _process_image_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Format shape and type of input image-feature tensor.
        """
        if features.dim() == 4:
            features = features[0, :, 0, 0]
        assert features.size() == (self.image_features_dim,)
        if self.use_cuda:
            features = features.cuda()
        else:
            features = features.cpu()
        if self.opt.get('fp16'):
            features = features.half()
        else:
            features = features.float()

        return features
