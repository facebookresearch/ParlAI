#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Image+Seq2Seq Agent.
"""
import torch
from typing import Dict, List, Tuple

from .modules import ImageSeq2seqModel
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.torch_agent import Batch

# from parlai.utils.typing import Dict, List


TOKEN_IMAGE = '__image__'
TOKEN_NO_IMAGE = '__no_image__'


class ImageSeq2seqAgent(TransformerGeneratorAgent):
    """
    ImageSeq2seqAgent Agent.

    Combines a transformer generator with images.
    """

    def build_model(self) -> ImageSeq2seqModel:
        """
        Override to build appropriate model.
        """
        self.model = ImageSeq2seqModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.embeddings.weight, self.opt['embedding_type']
            )
        return self.model

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Override to add one arg.
        """
        super(ImageSeq2seqAgent, cls).add_cmdline_args(argparser)
        group = argparser.add_argument_group('Image Encoder Args')
        group.add_argument(
            '--image-features-dim', type=int, default=2048, help='dim for image feats'
        )
        group.add_argument(
            '--image-encoder-num-layers',
            type=int,
            default=1,
            recommended=1,
            help='Number of layers for image encoder',
        )
        group.add_argument(
            '--include-image-token',
            type='bool',
            default=True,
            recommended=True,
            help='if true, include image token (or no image token) for each example',
        )

    def build_dictionary(self) -> DictionaryAgent:
        """
        Override to include image tokens.
        """
        self.dict = super().build_dictionary()
        if self.opt.get('include_image_token') and TOKEN_IMAGE not in self.dict:
            self.dict[TOKEN_IMAGE] = 1
            self.dict[TOKEN_NO_IMAGE] = 1

        return self.dict

    def _set_text_vec(self, *args, **kwargs) -> dict:
        """
        Override to include image token.
        """
        obs = super()._set_text_vec(*args, **kwargs)
        if 'text' not in obs or 'text_vec' not in obs:
            return obs
        if self.opt.get('include_image_token', False):
            # `truncate` is the third arg to this function
            truncate = args[2] - 1 if args[2] is not None else None
            vec = torch.LongTensor(
                self._check_truncate(obs['text_vec'], truncate, True)
            )
            token = TOKEN_NO_IMAGE
            if obs.get('image', None) is not None:
                token = TOKEN_IMAGE
            obs.force_set(
                'text_vec',
                torch.cat([vec, vec.new_tensor(self.dict[token]).unsqueeze(0)], 0),
            )
        return obs

    def _dummy_batch(self, batchsize: int, maxlen: int) -> Batch:
        """
        Override to include image feats.
        """
        return Batch(
            text_vec=torch.ones(batchsize, maxlen).long().cuda(),
            label_vec=torch.ones(batchsize, 2).long().cuda(),
            image=torch.ones(batchsize, self.opt.get('image_features_dim')).cuda(),
            personalities=torch.ones(batchsize, self.opt.get('embedding_size')).cuda(),
        )

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        """
        Override to handle images.
        """
        batch = super().batchify(obs_batch, sort)

        def _process_img(img):
            if img is not None and isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img[0, :, 0, 0]
                if self.use_cuda:
                    img = img.cuda()
                if self.opt.get('fp16'):
                    img = img.half()
                else:
                    img = img.float()

            return img

        if type(batch.image) == list and any(b is not None for b in batch):
            images = []
            for img in batch.image:
                images.append(_process_img(img))
            batch.image = images
        return batch

    def _model_input(self, batch: Batch) -> Tuple[torch.Tensor, List[object]]:
        return (batch.text_vec, batch.image)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Override for custom loading.

        Three reasons:
            1. When using an init model without an image encoder
            2. When using an init model with only an encoder provided
                In this case, we may need to add the START token to the state_dict
            3. When using an init model without image tokens in the embeddings.
                This is only the case if the embs differ by 2 in dimension 0
        """
        # Case 1 -> No Image Encoder
        if 'encoder.image_encoder.0.weight' not in state_dict:
            for k, v in self.model.encoder.image_encoder.state_dict().items():
                state_dict[f'encoder.image_encoder.{k}'] = v

        # Case 2 -> Only an Encoder provided
        if not (any('decoder' in state_key for state_key in state_dict)):
            for k, v in self.model.decoder.state_dict().items():
                state_dict[f'decoder.{k}'] = v
            state_dict['decoder.embeddings.weight'] = state_dict['embeddings.weight']
            if 'START' not in state_dict:
                state_dict['START'] = self.model.START

        if self.opt['init_model'] is not None:
            try:
                self.model.load_state_dict(state_dict)
                return
            except RuntimeError as e:
                # Case 3 --> Check for Embedding Diffs. Make sure dims match up
                embs = state_dict['embeddings.weight']
                enc_embs = state_dict['encoder.embeddings.weight']
                dec_embs = state_dict['decoder.embeddings.weight']
                init_embs = self.model.embeddings.weight
                if (
                    embs.shape[0] + 2 != init_embs.shape[0]
                    or embs.shape[1] != init_embs.shape[1]
                ):
                    raise e

                state_dict.update(
                    {
                        'embeddings.weight': torch.cat(
                            (
                                embs.to(init_embs.device, dtype=init_embs.dtype),
                                init_embs[-2:, :],
                            )
                        ),
                        'encoder.embeddings.weight': torch.cat(
                            (
                                enc_embs.to(init_embs.device, dtype=init_embs.dtype),
                                init_embs[-2:, :],
                            )
                        ),
                        'decoder.embeddings.weight': torch.cat(
                            (
                                dec_embs.to(init_embs.device, dtype=init_embs.dtype),
                                init_embs[-2:, :],
                            )
                        ),
                    }
                )
                pct_init = round(embs.shape[0] / len(self.dict) * 100, 1)
                print(
                    f'Initialized embeddings for {embs.shape[0]} tokens ({pct_init}%)'
                )

        self.model.load_state_dict(state_dict)
