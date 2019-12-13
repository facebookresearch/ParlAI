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
from parlai.core.torch_agent import Batch, Output
from parlai.utils.misc import round_sigfigs, Opt

try:
    from nltk.translate import bleu_score as nltkbleu

except ImportError:
    nltkbleu = None

try:
    from fairseq import bleu as fairseq_bleu

except ImportError:
    fairseq_bleu = None

TOKEN_IMAGE = '__image__'
TOKEN_NO_IMAGE = '__no_image__'


class ImageSeq2seqAgent(TransformerGeneratorAgent):
    """
    ImageSeq2seqAgent Agent.

    Combines a transformer generator with images.
    """

    def __init__(self, opt: Opt, shared: dict = None):
        super().__init__(opt, shared)
        self.compute_tokenized_bleu = opt.get('compute_tokenized_bleu', False)
        if not shared:
            self._init_bleu_scorers()
        else:
            self.fairseq_bleu_scorer = shared['fairseq_bleu_scorer']
            self.nltk_bleu = shared['nltk_bleu']
            self.nltk_bleu_cnts = shared['nltk_bleu_cnts']

    def _init_bleu_scorers(self):
        if not hasattr(self, 'fairseq_bleu_scorer'):
            if fairseq_bleu is None:
                self.fairseq_bleu_scorer = None
            else:
                self.fairseq_bleu_scorer = fairseq_bleu.Scorer(
                    self.NULL_IDX, self.END_IDX, self.dict[self.dict.unk_token]
                )
        self.nltk_bleu = {f'bleu-{i}': 0 for i in range(1, 5)}
        self.nltk_bleu_cnts = {f'bleu-{i}': 0 for i in range(1, 5)}

    def share(self) -> dict:
        """
        Override to include BLEU scorer.
        """
        shared = super().share()
        shared['fairseq_bleu_scorer'] = self.fairseq_bleu_scorer
        shared['nltk_bleu'] = self.nltk_bleu
        shared['nltk_bleu_cnts'] = self.nltk_bleu_cnts
        return shared

    def build_model(self) -> ImageSeq2seqModel:
        """
        Override to build appropriate model.
        """
        self.model = ImageSeq2seqModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model = self.model.cuda()

        return self.model

    def reset_metrics(self):
        super().reset_metrics()
        self._init_bleu_scorers()

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
        group.add_argument(
            '--compute-tokenized-bleu',
            type='bool',
            default=False,
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
            3. When using an init model without image tokens in the embeddings
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

        cnt = 0
        if self.opt['init_model'] is not None:
            try:
                self.model.load_state_dict(state_dict)
                return
            except Exception as e:
                # Case 3 --> Check for Embedding Diffs
                print('exception in loading state dict: {}'.format(e))
                # make sure dims match up
                embs = state_dict['embeddings.weight']
                enc_embs = state_dict['encoder.embeddings.weight']
                dec_embs = state_dict['decoder.embeddings.weight']
                num_embs = min(self.model.embeddings.weight.shape[0], embs.shape[0])
                num_enc_embs = min(
                    self.model.encoder.embeddings.weight.shape[0], enc_embs.shape[0]
                )
                num_dec_embs = min(
                    self.model.decoder.embeddings.weight.shape[0], dec_embs.shape[0]
                )
                for i in range(num_embs):
                    self.model.embeddings.weight.data[i] = embs[i]
                    cnt += 1
                for i in range(num_enc_embs):
                    self.model.encoder.embeddings.weight.data[i] = enc_embs[i]
                for i in range(num_dec_embs):
                    self.model.decoder.embeddings.weight.data[i] = dec_embs[i]
                state_dict['embeddings.weight'] = self.model.embeddings.weight
                state_dict[
                    'encoder.embeddings.weight'
                ] = self.model.encoder.embeddings.weight
                state_dict[
                    'decoder.embeddings.weight'
                ] = self.model.decoder.embeddings.weight

                print(
                    'Initialized embeddings for {} tokens ({}%).'
                    ''.format(cnt, round(cnt * 100 / len(self.dict), 1))
                )
        self.model.load_state_dict(state_dict)

    def eval_step(self, batch: Batch) -> Output:
        """
        Evaluate a single batch of examples.

        Override to compute correct BLEU scores.
        """
        output = super().eval_step(batch)
        if output is None or (self.skip_generation and not self.compute_tokenized_bleu):
            return output

        texts = output.text
        self._compute_fairseq_bleu(batch, texts)
        self._compute_nltk_bleu(batch, texts)

        return output

    def report(self) -> dict:
        """
        Override to include custom BLEU computations.
        """
        metrics = super().report()
        metrics.update(
            {
                'fairseq_bleu': 'N/A',
                'nltk_bleu_unnormalized': 'N/A'
            }
        )
        if not self.skip_generation and self.compute_tokenized_bleu:
            if fairseq_bleu is not None:
                try:
                    fairseq_bleu_scores = {
                        k: self.fairseq_bleu_scorer.result_string(order=k)
                        for k in range(1, 5)
                    }
                except ZeroDivisionError:
                    # some preds are REAL bad
                    fairseq_bleu_scores = {k: 0 for k in range(1, 5)}

                metrics['fairseq_bleu'] = {
                    k: v[v.index('= ') : v.index(',')]
                    for k, v in fairseq_bleu_scores.items()
                }
            if nltkbleu is not None:
                metrics['nltk_bleu_unnormalized'] = {
                    k: round_sigfigs(v / self.nltk_bleu_cnts[k], 4)
                    for k, v in self.nltk_bleu.items()
                }
        return metrics

    def _compute_fairseq_bleu(self, batch: Batch, texts: List[str]):
        """
        Compute BLEU score between text and label, using the FAIRSeq BLEU Scorer.

        :param batch:
        """
        if fairseq_bleu is None:
            return 0
        aa = torch.IntTensor(1)
        for i, t in enumerate(texts):
            self.fairseq_bleu_scorer.add(
                batch.label_vec[i].type_as(aa),
                self._vectorize_text(t, True, True, self.label_truncate, False).type_as(
                    aa
                ),
            )

    def _compute_nltk_bleu(self, batch: Batch, texts: List[str]):
        def _bleu(guess: str, answers: List[str], weights: List[float]):
            """
            Compute approximate BLEU score between guess and a set of answers.

            This function does not process guess or answers, as opposed to the normal
            bleu function in metrics.py
            """
            if nltkbleu is None:
                return 0
            return nltkbleu.sentence_bleu(
                [a.split(" ") for a in answers],
                guess.split(" "),
                smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
                weights=weights,
            )

        for i, p in enumerate(texts):
            obs = batch.observations[i]
            references = []
            for lbl in obs['eval_labels']:
                references.append(
                    self._v2t(
                        self._vectorize_text(
                            lbl, True, True, self.label_truncate, False
                        )
                    )
                )
            for i in range(4):
                weights = [1 / (i + 1) for _ in range(i + 1)]
                self.nltk_bleu[f'bleu-{i + 1}'] += _bleu(p, references, weights)
                self.nltk_bleu_cnts[f'bleu-{i + 1}'] += 1
