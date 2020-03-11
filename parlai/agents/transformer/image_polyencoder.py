#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure -m transformer/generator works as expected
"""
Poly-encoder agent that ingests image features.
"""

from typing import Any, Dict

import torch

from parlai.agents.image_seq2seq.modules import ContextWithImageEncoder
from parlai.agents.transformer.modules import get_n_positions_from_options
from parlai.agents.transformer.polyencoder import PolyencoderAgent, PolyEncoderModule
from parlai.core.torch_agent import Batch
from parlai.core.torch_image_agent import TorchImageAgent
from parlai.utils.misc import warn_once


class ImagePolyencoderAgent(PolyencoderAgent, TorchImageAgent):
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
        PolyencoderAgent.add_cmdline_args(argparser)
        TorchImageAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('ImagePolyencoder Args')
        agent.add_argument(
            '--image-combination-mode',
            type=str,
            default='prepend',
            choices=['add', 'append', 'prepend'],
            help='How to combine image embedding (if used) with context embedding',
        )
        # TODO: more thoroughly test out whether one of these choices is best and add a
        #  'recommended' arg here. 'add' and 'prepend' seem to be roughly similar in
        #  performance
        agent.add_argument(
            '--n-image-tokens',
            type=int,
            default=1,
            help=(
                'Number of tokens that the image encoding will consist of (when adding '
                'or prepending)'
            ),
        )
        agent.set_defaults(reduction_type=None)
        # This agent doesn't support any encoder output reductions
        return agent

    def build_model(self, states=None):
        """
        Return built model.
        """
        return ImagePolyencoderModule(self.opt, self.dict, self.NULL_IDX)

    def batchify_image_features(self, batch: Batch) -> Batch:
        """
        Return the image features as a Tensor of the correct type.

        Fill in missing feature vectors. Here, we require image features to be saved in
        `batch` as a Tensor for passing through the image encoder. This is required for
        data_parallel.
        """

        # Checks/formatting of batch.image
        bsz = self._get_batch_size(batch)
        if batch.image is None or len(batch.image) == 0:
            batch.image = [None] * bsz
        else:
            assert len(batch.image) == bsz

        # Process all image feature vectors, or add in zero vectors if missing
        processed_features_list = []
        processed_zero_features = self._process_image_features(
            torch.zeros((self.image_features_dim,))
        )
        for orig_features in batch.image:
            if isinstance(orig_features, torch.Tensor):
                processed_features_list.append(
                    self._process_image_features(orig_features)
                )
            else:
                if orig_features is not None:
                    warn_once(
                        'Unsupported image feature format. Image features will be ignored!'
                    )
                processed_features_list.append(processed_zero_features)

        # Turn into batchsize x image_features_dim for DataParallel
        batch.image = torch.stack(processed_features_list)

        return batch

    def _get_batch_size(self, batch) -> int:
        """
        Return the size of the batch.

        Use the size of the text vec if it exists; otherwise, use the length of the
        image feature list.
        """
        if batch.text_vec is not None:
            return batch.text_vec.size(0)
        else:
            return len(batch.image)

    def _model_context_input(self, batch) -> Dict[str, Any]:
        """
        Override PolyencoderAgent's context inputs into the model.
        """
        return {'ctxt_tokens': batch.text_vec, 'ctxt_image': batch.image}

    def load_state_dict(self, state_dict):
        """
        Override to account for weights used for image features.
        """
        for tensor in ['dummy_image_enc', 'ones_mask']:
            key = f'encoder_ctxt.{tensor}'
            if hasattr(self.model.encoder_ctxt, tensor) and key not in state_dict:
                state_dict[key] = getattr(self.model.encoder_ctxt, tensor)
        if hasattr(self.model.encoder_ctxt, 'image_encoder'):
            for layer_idx, layer in enumerate(self.model.encoder_ctxt.image_encoder):
                for tensor in ['weight', 'bias']:
                    key = f'encoder_ctxt.image_encoder.{layer_idx}.{tensor}'
                    if hasattr(layer, tensor) and key not in state_dict:
                        state_dict[key] = getattr(layer, tensor)
        super().load_state_dict(state_dict)


class ImagePolyencoderModule(PolyEncoderModule):
    """
    Poly-encoder model with image features.

    Model that allows encoding image features and adding or concatenating them to the
    context encoding.
    """

    def get_encoder(self, opt, dict_, null_idx, reduction_type, for_context: bool):
        """
        Return encoder that allows for image features to be passed in, given options.

        :param opt:
            opt dict
        :param dict:
            dictionary agent
        :param null_idx:
            null/pad index into dict
        :param reduction_type: only used for compatibility with the superclass method
        :param for_context:
            whether this is the context encoder (as opposed to the candidate encoder)
        :return:
            either a TransformerEncoder or a ContextWithImageEncoder, initialized
            correctly
        """
        if for_context:
            if reduction_type is not None:
                raise NotImplementedError('No encoder output reductions supported!')
            n_positions = get_n_positions_from_options(opt)
            embeddings = self._get_embeddings(
                dict_=dict_, null_idx=null_idx, embedding_size=opt['embedding_size']
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
                n_segments=opt['n_segments'],
                activation=opt['activation'],
                variant=opt['variant'],
                output_scaling=opt['output_scaling'],
                image_encoder_num_layers=opt['image_encoder_num_layers'],
                image_features_dim=opt['image_features_dim'],
                image_combination_mode=opt['image_combination_mode'],
                n_image_tokens=opt['n_image_tokens'],
            )
        else:
            # The candidate encoder is the same as for PolyEncoderModule
            return super().get_encoder(
                opt=opt,
                dict_=dict_,
                null_idx=null_idx,
                reduction_type=reduction_type,
                for_context=for_context,
            )

    def _context_encoder_input(self, ctxt_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override PolyEncoderModule's inputs into the context encoder.
        """
        assert set(ctxt_inputs.keys()) == {'ctxt_tokens', 'ctxt_image'}
        return {
            'src_tokens': ctxt_inputs['ctxt_tokens'],
            'image_features': ctxt_inputs['ctxt_image'],
        }

    def _get_context_batch_size(self, **ctxt_inputs: torch.Tensor) -> int:
        """
        Return the batch size of the context.
        """
        if ctxt_inputs['ctxt_tokens'] is not None:
            return ctxt_inputs['ctxt_tokens'].size(0)
        else:
            return ctxt_inputs['ctxt_image'].size(0)
