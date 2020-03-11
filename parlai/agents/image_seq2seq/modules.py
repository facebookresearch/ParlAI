#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Modules for ImageSeq2seqAgent Agent.
"""

from functools import reduce
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn

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

    Encodes image features and context, and combines by summing or concatenation.
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
        n_positions=1024,
        activation='relu',
        variant='aiayn',
        n_segments=0,
        output_scaling=1.0,
        image_encoder_num_layers=1,
        image_features_dim=2048,
        image_combination_mode='append',
        n_image_tokens=1,
    ):
        """
        Override TransformerEncoder __init__.

        Setup the image encoder; create some dummy tensors for inserting image into
        input
        """

        self.padding_idx = padding_idx
        self.n_img_layers = image_encoder_num_layers
        self.img_dim = image_features_dim
        self.image_combination_mode = image_combination_mode
        self.n_image_tokens = n_image_tokens
        if self.image_combination_mode == 'add' and self.n_image_tokens > 1:
            raise ValueError(
                'Image encoding cannot be added to context encoding if there is more than one image token!'
            )
        reduction_type = None  # Must pass back unreduced encoding and mask
        super().__init__(
            n_heads=n_heads,
            n_layers=n_layers,
            embedding_size=embedding_size,
            ffn_size=ffn_size,
            vocabulary_size=vocabulary_size,
            embedding=embedding,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            padding_idx=padding_idx,
            learn_positional_embeddings=learn_positional_embeddings,
            embeddings_scale=embeddings_scale,
            reduction_type=reduction_type,
            n_positions=n_positions,
            activation=activation,
            variant=variant,
            n_segments=n_segments,
            output_scaling=output_scaling,
        )
        self.full_embedding_size = self.embedding_size * self.n_image_tokens
        # Images will be embedded to this size, and then the embedding will be folded
        # into however many tokens are needed
        self._build_image_encoder()
        self.register_buffer(
            'dummy_image_enc', torch.zeros((self.full_embedding_size,))
        )
        self.register_buffer('ones_mask', torch.ones(self.n_image_tokens).bool())

    def _build_image_encoder(self):
        image_layers = [nn.Linear(self.img_dim, self.full_embedding_size)]
        for _ in range(self.n_img_layers - 1):
            image_layers += [
                nn.ReLU(),
                nn.Dropout(p=self.dropout_frac),
                nn.Linear(self.full_embedding_size, self.full_embedding_size),
            ]
        self.image_encoder = nn.Sequential(*image_layers)

    def encode_images(
        self, images: Union[List[object], torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode Images.

        Encodes images given in `images`, if the image can be encoded (i.e. it
        is a tensor).

        :param images:
            either a list of objects of length N, of which some maybe be None, or a
            tensor of shape (batch size, self.img_dim)

        :return:
            a (image_encoded, image_mask) tuple, where:

            - image_encoded is a torch.Tensor of dim N x self.n_image_tokens x
              self.embedding_size, representing the encoded batch of images
            - image_mask is a torch.Tensor of dim N x self.n_image_tokens
        """
        image_masks = image_encoded = None
        valid_inds = [
            i
            for i, img in enumerate(images)
            if img is not None and isinstance(img, torch.Tensor)
        ]

        if valid_inds:
            image_mask_list = []
            image_encoded_list = []

            valid_imgs = torch.stack([images[i] for i in valid_inds])
            valid_img_enc = self.image_encoder(valid_imgs)

            img_num = 0
            for i in range(len(images)):
                if i in valid_inds:
                    image_mask_list.append(self.ones_mask)
                    image_encoded_list.append(valid_img_enc[img_num, :])
                    img_num += 1
                else:
                    image_mask_list.append(~self.ones_mask)
                    image_encoded_list.append(self.dummy_image_enc)

            image_masks = torch.stack(image_mask_list)
            image_encoded = torch.stack(image_encoded_list).reshape(
                (len(images), self.n_image_tokens, self.embedding_size)
            )
            assert image_masks.shape == image_encoded.shape[:2]

        return image_encoded, image_masks

    def forward(
        self,
        src_tokens: Optional[torch.Tensor],
        image_features: Optional[Union[List[object], torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode images with context.

        Encodes tokens (if given) and images (if given) separately.
        Combines via either addition, prepending, or appending the image embedding to
        the context embedding.

        :param src_tokens:
            A bsz x seq_len tensor of src_tokens; possibly None
        :param image_features:
            Either a list of (torch.tensor) or a tensor of shape (batch_size,
            self.img_dim)

        :return:
            A (full_enc, full_mask) tuple, which represents the encoded context
            and the mask
        """
        context_encoded = context_mask = None
        image_encoded = extra_masks = None
        if src_tokens is not None and image_features is not None:
            assert src_tokens.size(0) == len(image_features)
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

        if self.image_combination_mode == 'add':
            full_enc = self._add([context_encoded, image_encoded])
            # image_encoded broadcasted along dim=1
            full_mask = context_mask
        elif self.image_combination_mode == 'append':
            full_enc = self._cat([context_encoded, image_encoded])
            full_mask = self._cat([context_mask, extra_masks])
        elif self.image_combination_mode == 'prepend':
            full_enc = self._cat([image_encoded, context_encoded])
            full_mask = self._cat([extra_masks, context_mask])
        else:
            raise ValueError('Image combination mode not recognized!')

        if full_enc.dtype == torch.half:
            full_enc, full_mask = self._fix_for_fp16(
                full_enc=full_enc, full_mask=full_mask
            )

        return full_enc, full_mask

    def _add(self, tensors: List[Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Handle addition of None tensors.

        Smart addition. Adds tensors if they are not None.

        :param tensors:
            A list of torch.Tensor, with at least one non-null object

        :return:
            The result of adding all non-null objects in tensors
        """
        tensors = [t for t in tensors if t is not None]
        return reduce(lambda a, b: a + b, tensors)

    def _cat(self, tensors: List[Optional[torch.Tensor]]) -> torch.Tensor:
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

    def _fix_for_fp16(
        self, full_enc: torch.Tensor, full_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        In fp16 mode, either remove extra tokens or add new ones on to get to a multiple
        of 8.
        """

        if full_mask is None:
            # full_mask is None corresponds to no input tokens, and in case there are no
            # tokens to add/remove to get a multiple of 8
            return full_enc, full_mask

        num_tokens_to_remove = full_enc.size(1) % 8
        if num_tokens_to_remove == 0:
            # Tensor already divisible by 8
            pass
        elif (~full_mask[:, -num_tokens_to_remove:].all()).item():
            # The tokens we'd like to remove are all padding, so subtract them from
            # the end
            full_enc = full_enc[:, :-1, :]
            full_mask = full_mask[:, :-1]
        else:
            # We can't subtract that many padding tokens, so add some to the end
            num_tokens_to_add = 8 - num_tokens_to_remove
            enc_extension = full_enc.new_full(
                size=(full_enc.size(0), num_tokens_to_add, full_enc.size(2)),
                fill_value=self.padding_idx,
            )
            mask_extension = full_mask.new_full(
                size=(full_mask.size(0), num_tokens_to_add), fill_value=self.padding_idx
            )
            full_enc = torch.cat([full_enc, enc_extension], dim=1)
            full_mask = torch.cat([full_mask, mask_extension], dim=1)
        return full_enc, full_mask
