#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Modules for ImageSeq2seqAgent Agent.
"""
import torch
import torch.nn as nn
from typing import List, Tuple

from parlai.agents.transformer.modules import (
    TransformerGeneratorModel,
    TransformerEncoder,
)
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt


class ImageSeq2seqModel(TransformerGeneratorModel):
    """
    ImageSeq2seqModel.

    Just TGA that can encode image with encoder.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent):
        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0,
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        super().__init__(opt, dictionary)
        self.encoder = ContextWithImageEncoder(
            n_heads=opt['n_heads'],
            n_layers=opt['n_layers'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dictionary),
            embedding=self.embeddings,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=self.pad_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            reduction_type=None,
            n_positions=n_positions,
            n_segments=opt.get('n_segments', 0),
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
            image_encoder_num_layers=opt['image_encoder_num_layers'],
            image_features_dim=opt['image_features_dim'],
        )


class ContextWithImageEncoder(TransformerEncoder):
    """
    ContextWithImage Module.

    Encodes image and context via simple concatenation.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction_type='mean',
        n_positions=1024,
        activation='relu',
        variant='aiayn',
        n_segments=0,
        output_scaling=1.0,
        image_encoder_num_layers=1,
        image_features_dim=2048,
    ):
        """
        Override TransformerEncoder __init__.

        Setup the image encoder; create some dummy tensors for inserting image into
        input
        """
        self.n_img_layers = image_encoder_num_layers
        self.img_dim = image_features_dim
        super().__init__(
            n_heads,
            n_layers,
            embedding_size,
            ffn_size,
            vocabulary_size,
            embedding,
            dropout,
            attention_dropout,
            relu_dropout,
            padding_idx,
            learn_positional_embeddings,
            embeddings_scale,
            reduction_type,
            n_positions,
            activation,
            variant,
            n_segments,
            output_scaling,
        )
        self._build_image_encoder()
        self.dummy_image_enc = torch.nn.Parameter(
            torch.zeros((self.embedding_size)), requires_grad=False
        )
        self.ones_mask = torch.nn.Parameter(torch.ones(1).bool(), requires_grad=False)

    def _build_image_encoder(self):
        image_layers = [nn.Linear(self.img_dim, self.embedding_size)]
        for _ in range(self.n_img_layers - 1):
            image_layers += [
                nn.ReLU(),
                nn.Dropout(p=self.opt['dropout']),
                nn.Linear(self.img_dim, self.embedding_size),
            ]
        self.image_encoder = nn.Sequential(*image_layers)

    def encode_images(self, images: List[object]) -> Tuple[List[int], torch.Tensor]:
        """
        Encode Images.

        Encodes images given in `images`, if the image can be encoded (i.e. it
        is a tensor).

        :param images:
            list of objects of length N, of which some maybe be None

        :return:
            a (image_encoded, image_mask) tuple, where:

            - image_enc is a torch.Tensor of dim N x self.img_dim,
              representing the encoded batch of images
            - image_mask is a torch.Tensor of dim N x 1
        """
        image_masks = image_encoded = None
        valid_inds = [
            i
            for i, img in enumerate(images)
            if img is not None and isinstance(img, torch.Tensor)
        ]

        if valid_inds:
            image_masks = []
            image_encoded = []

            valid_imgs = torch.stack([images[i] for i in valid_inds])
            valid_img_enc = self.image_encoder(valid_imgs)

            img_num = 0
            for i in range(len(images)):
                if i in valid_inds:
                    image_masks.append(self.ones_mask)
                    image_encoded.append(valid_img_enc[img_num, :])
                    img_num += 1
                else:
                    image_masks.append(~self.ones_mask)
                    image_encoded.append(self.dummy_image_enc)

            image_masks = torch.stack(image_masks)
            image_encoded = torch.stack(image_encoded).unsqueeze(1)

        return image_encoded, image_masks

    def forward(
        self, src_tokens: torch.Tensor, image_features: List[object]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images with context.

        Encodes tokens (if given) and images (if given) separately.
        Combines via concatenation, where images are added to the end of the tensor.

        :param src_tokens:
            A bsz x seq_len tensor of src_tokens; possibly None
        :param image_features:
            A list of (torch.tensor)

        :return:
            A (full_enc, full_mask) tuple, which represents the encoded context
            and the mask
        """
        context_encoded = context_mask = None
        image_encoded = extra_masks = None
        if src_tokens is not None:
            context_encoded, context_mask = super().forward(src_tokens)
        if image_features is not None:
            image_encoded, extra_masks = self.encode_images(image_features)

        if all(enc is None for enc in [context_encoded, image_encoded]):
            raise RuntimeError(
                'You are providing Image+Seq2Seq with no input.\n'
                'If you are using a text-based task, make sure the first turn '
                'has text (e.g. a __SILENCE__ token if the model starts the convo).\n'
                'If you are using an image-based task, make sure --image-mode is '
                'set correctly.'
            )

        full_enc = self.cat([context_encoded, image_encoded])
        full_mask = self.cat([context_mask, extra_masks])
        return full_enc, full_mask

    def cat(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Handle concatenation of None tensors.

        Smart concatenation. Concatenates tensors if they are not None.

        :param tensors:
            A list of torch.Tensor, with at least one non-null object

        :return:
            The result of concatenating all non-null objects in tensors
        """
        tensors = [t for t in tensors if t is not None]
        return torch.cat([t for t in tensors], dim=1)
