#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure -m transformer/generator works as expected
"""
Poly-encoder agent that ingests image features.
"""
from typing import List, Dict, Any

import torch

from parlai.agents.image_seq2seq.modules import ContextWithImageEncoder
from parlai.agents.transformer.modules import get_n_positions_from_options
from parlai.agents.transformer.polyencoder import PolyencoderAgent, PolyEncoderModule
from parlai.core.message import Message
from parlai.core.torch_agent import Batch

DEFAULT_IMAGE_FEATURES_DIM = 2048


class ImagePolyencoderAgent(PolyencoderAgent):
    """
    Poly-encoder Agent that ingests image features.

    Agent that allows encoding image features and adding or concatenating them to the
    context encoding.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        super(ImagePolyencoderAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Image Encoder Args')
        agent.add_argument(
            '--polyencoder-image-encoder-num-layers',
            type=int,
            default=1,
            help='Number of linear layers to encode image features with in the context',
        )
        agent.add_argument(
            '--polyencoder-image-features-dim',
            type=int,
            default=DEFAULT_IMAGE_FEATURES_DIM,
            help='For passing in image features of the given dim in the context',
        )
        agent.add_argument(
            '--polyencoder-image-combination-mode',
            type=str,
            default='prepend',
            choices=['add', 'append', 'prepend'],
            help='How to combine image embedding (if used) with context embedding',
        )
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.image_features_dim = opt.get(
            'polyencoder_image_features_dim', DEFAULT_IMAGE_FEATURES_DIM
        )

    def build_model(self, states=None):
        """
        Return built model.
        """
        return ImagePolyencoderModule(self.opt, self.dict, self.NULL_IDX)

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        """
        Override to handle image features.
        """
        batch = super().batchify(obs_batch, sort)

        def _process_image_features(features: torch.Tensor) -> torch.Tensor:
            assert features.size() == (self.image_features_dim,)
            if self.use_cuda:
                features = features.cuda()
            if self.opt.get('fp16'):
                features = features.half()
            else:
                features = features.float()

            return features

        # Checks/formatting of batch.image
        bsz = batch.text_vec.size(0)
        if batch.image is None or len(batch.image) == 0:
            batch.image = [None] * bsz
        else:
            assert len(batch.image) == bsz

        # Process all image feature vectors, or add in zero vectors if missing
        processed_features_list = []
        processed_zero_features = _process_image_features(
            torch.zeros((self.image_features_dim,))
        )
        for orig_features in batch.image:
            if orig_features is None:
                processed_features_list.append(processed_zero_features)
            elif isinstance(orig_features, torch.Tensor):
                processed_features_list.append(_process_image_features(orig_features))
            else:
                raise ValueError('Unsupported image feature format!')

        # Turn into batchsize x polyencoder_image_features_dim for DataParallel
        batch.image = torch.stack(processed_features_list)

        return batch

    def _model_context_input(self, batch) -> tuple:
        """Override PolyencoderAgent's context inputs into the model."""
        return (batch.text_vec, batch.image)

    def load_state_dict(self, state_dict):
        """
        Override to account for weights used for image features.
        """
        for tensor in ['dummy_image_enc', 'ones_mask']:
            key = f'encoder_ctxt.{tensor}'
            val = getattr(self.model.encoder_ctxt, tensor, None)
            if val is not None and key not in state_dict:
                state_dict[key] = val
        if hasattr(self.model.encoder_ctxt, 'image_encoder'):
            for tensor in ['weight', 'bias']:
                key = f'encoder_ctxt.image_encoder.0.{tensor}'
                val = getattr(self.model.encoder_ctxt.image_encoder[0], tensor, None)
                if val is not None and key not in state_dict:
                    state_dict[key] = val
        super().load_state_dict(state_dict)


class ImagePolyencoderModule(PolyEncoderModule):
    """
    Poly-encoder model with image features.

    Model that allows encoding image features and adding or concatenating them to the
    context encoding.
    """

    def __init__(self, opt, dict_, null_idx):
        super().__init__(opt=opt, dict=dict_, null_idx=null_idx)
        self.encoder_ctxt = self.get_encoder(opt=opt, dict_=dict_, null_idx=null_idx)

    def get_encoder(self, opt, dict_, null_idx):
        """
        Return encoder that allows for image features to be passed in, given options.

        :param opt:
            opt dict
        :param dict_:
            dictionary agent
        :param null_idx:
            null/pad index into dict
        :return:
            a ContextWithImageEncoder, initialized correctly
        """
        n_positions = get_n_positions_from_options(opt)
        embeddings = self._get_embeddings(
            dict=dict_, null_idx=null_idx, embedding_size=opt['embedding_size']
        )
        return ContextWithImageEncoder(
            n_heads=opt['n_heads'],
            n_layers=opt['n_layers'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dict_),
            embedding=embeddings,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=null_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            n_positions=n_positions,
            n_segments=2,
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
            image_encoder_num_layers=opt['polyencoder_image_encoder_num_layers'],
            image_features_dim=opt['polyencoder_image_features_dim'],
            image_combination_mode=opt['polyencoder_image_combination_mode'],
        )
