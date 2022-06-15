#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DirectorAgent for Supervised Language Modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Dict, Tuple, Union

from parlai.agents.transformer.modules import TransformerGeneratorModel
from parlai.agents.transformer.transformer import TransformerGeneratorAgent

from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import PPLMetric

import parlai.utils.logging as logging


class DirectorModel(TransformerGeneratorModel):
    """
    Director model that extends TransformerGeneratorModel and adds |V| binary classifier
    heads.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, **kwargs):
        super().__init__(opt, dictionary, **kwargs)

        vocabulary_size = len(dictionary)

        decoder_output_dim = self.decoder.out_dim
        self.classifier_heads = nn.Linear(decoder_output_dim, vocabulary_size)

        self.infer_gamma = opt['train_gamma']
        if opt.get('infer_gamma') is not None:
            self.infer_gamma = opt['infer_gamma']

        self.freeze_decoder = opt['freeze_decoder']

    def generator_output(self, input: torch.Tensor):
        if self.freeze_decoder:
            input = input.detach()

        return super().output(input)

    def classifier_output(self, input: torch.Tensor):
        if self.freeze_decoder:
            input = input.detach()

        return self.classifier_heads(input)

    def output(self, latent: torch.Tensor):
        """Overriding output method to use |V| classifier heads to modify the generator logprobs.
            This modification allows model to incorporate attribute information from classifier for selecting the next tokens.
        Args:
            latent (torch.Tensor): decoder outputs

        Returns:
            Modified logprobs.
        """
        classifier_outputs = F.logsigmoid(self.classifier_output(latent))
        log_predictor_scores = F.log_softmax(self.generator_output(latent), dim=-1)

        scores = log_predictor_scores + self.infer_gamma * classifier_outputs

        return F.log_softmax(scores, dim=-1)

    def load_state_dict(self, state_dict):
        """
        Overrided to load only the generator weights from the state dict and leaving the
        classifier head weights untouched.
        """
        for k, v in self.state_dict().items():
            if k not in state_dict:
                state_dict[k] = v

        super().load_state_dict(state_dict)

    def forward(
        self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.BoolTensor,
        Any,
    ]:
        """
        Nearly copied verbatim, except for return type to return the latent state and
        the classifier scores.
        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs)

        # use teacher forcing
        scores, preds, latent, mask = self.decode_forced(encoder_states, ys)

        classifer_score = self.classifier_output(latent)
        return scores, preds, classifer_score, latent, mask, encoder_states

    def decode_forced(
        self, encoder_states: Tuple[Any], ys: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.BoolTensor]:
        """
        Override TGM.decode_forced to return latent states and using generator_output
        method to generate the decoder output.
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        if (ys[:, 0] == self.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self._get_initial_forced_decoder_input(bsz, inputs)
        latent, mask = self.decoder(inputs, encoder_states)
        logits = self.generator_output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds, latent, mask


class DirectorAgent(TransformerGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        TransformerGeneratorAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        # This method will add arguments specific to fused classifier architecture
        # like num_layers classifier, etc.
        group = parser.add_argument_group('Director Group')
        group.add_argument(
            '--explicit-classifier-norm',
            type=bool,
            default=False,
            help='If we should explictly try to set non-target tokens to 0.5 during training.',
        )
        group.add_argument(
            '--explicit-classifier-norm-coeff',
            type=float,
            default=1,
            help='If we should explictly try to set non-target tokens to 0.5 during training.',
        )
        group.add_argument(
            '--freeze-decoder',
            type=bool,
            default=False,
            help='Freeze decoder for training only classifier head.',
        )
        group.add_argument(
            '--train-gamma',
            type=float,
            default=0.5,
            help="Implementing Sainaa's suggestion of keeping generator weight fixed (to 1) and using \alpha (hopefully <1) to weight classifier.",
        )
        group.add_argument(
            '--infer-gamma',
            type=float,
            default=None,
            help="Implementing Sainaa's suggestion of keeping generator weight fixed (to 1) and using \alpha (hopefully <1) to weight classifier.",
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.explicit_classifier_norm = opt['explicit_classifier_norm']
        self.explicit_classifier_norm_coeff = opt['explicit_classifier_norm_coeff']
        self.train_gamma = opt['train_gamma']
        self.infer_gamma = opt['infer_gamma']

        assert opt[
            'beam_block_full_context'
        ], 'must set --beam-block-full-context True to use PACER'

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This copies the classifier specific params to init_model and then calls the
        load_state_dict method of TorchAgent.
        """
        for k, v in self.model.state_dict().items():
            if k not in state_dict:
                state_dict[k] = v
        super().load_state_dict(state_dict)

    def _get_batch_context(self, batch):
        """
        Override to always provide full context.
        """
        if 'full_text_vec' not in batch:
            logging.warn('Batch does not have full text vec, resorting to text vec')
            return batch.text_vec
        return batch.full_text_vec

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = DirectorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def observe(self, observation: Union[Dict, Message]) -> Message:
        observation = super().observe(observation)
        if 'is_ltr' not in observation:
            observation['is_ltr'] = False
            observation['classifier_label'] = 'none'
            observation['classifier_label_idx'] = -1
            return observation

        classifier_label = observation['classifier_label']
        if classifier_label == 'pos':
            observation['classifier_label_idx'] = 1
        elif classifier_label == 'neg':
            observation['classifier_label_idx'] = 0
        return observation

    def batchify(self, obs_batch, sort=False):
        """
        This method calls the parent class's batchify method and then add
        classifier_label and is_ltr property to the the batch.
        """
        batch = super().batchify(obs_batch, sort=sort)

        if batch.valid_indices is None:
            return batch

        batch.classifier_label = torch.tensor(
            [
                [obs_batch[i].get('classifier_label_idx', -1)]
                for i in batch.valid_indices
            ]
        )
        batch.is_ltr = torch.tensor(
            [[obs_batch[i].get('is_ltr', False)] for i in batch.valid_indices]
        )
        return batch

    def _reshape_to_record_metrics(self, batch, losses, num_target_tokens, indices):
        """
        MultitaskAgent shuffles and combines examples from both classifier and the
        generator tasks in a single batch. We compute losses only for those exs in the
        batch resulting in losses and num_target_tokens vectors that are smaller than
        the.

        This method reshapes the losses and num_target_tokens vectors back to the batch size. This is needed to record local metrics as the metrics need to be of batch size.

        Args:
            batch: batch being processed in this iteration.
            losses: classifier or generator loss vector (shape: b' X 1), where b' <= b.
            num_target_tokens: number of tokens in each examples for classification or generation tasks. (shape: b' X 1), where b' <= b.
            indices: indices of (either classification or generation) exs for which the loss was computed.

        Returns:
            A tuple of reshaped losses and num_target_tokens, both of shape: b X 1.
        """
        val_id_shape = batch.valid_indices.shape
        reshaped_losses = torch.zeros(
            val_id_shape, device=losses.device, dtype=losses.dtype
        )
        reshaped_num_target_tokens = torch.zeros(
            val_id_shape, device=num_target_tokens.device, dtype=num_target_tokens.dtype
        )

        reshaped_losses[indices] = losses
        reshaped_num_target_tokens[indices] = num_target_tokens

        return (reshaped_losses, reshaped_num_target_tokens)

    def _v2t(self, vec):
        """
        This method is copied from TFGA but wraps the vec2txt call in a try catch to
        ensure that sequences with generation errors are ignored.

        We return a empty string instead in that scenario.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.END_IDX:
                break
            elif i != self.START_IDX:
                new_vec.append(i)

        try:
            txt = self.dict.vec2txt(new_vec)
        except AssertionError:
            txt = ""
        return txt

    def compute_classifier_loss(self, classifier_scores, batch):
        bsz = batch.batchsize
        device = classifier_scores.device

        classifier_losses = torch.zeros((bsz,), device=device)
        num_target_tokens = torch.zeros((bsz,), device=device, dtype=torch.long)

        # idxs of all the classification exs in the batch.
        classification_idxs = batch.is_ltr[:, 0]

        self.record_local_metric(
            'pct_classifier_exs', AverageMetric.many(classification_idxs)
        )

        # if none of the exs in the batch are classifier examples,
        # return zero classifier loss.
        if not torch.any(classification_idxs):
            return classifier_losses, num_target_tokens

        classifier_scores = classifier_scores[classification_idxs]

        # Select the classifier scores for next tokens
        target_tokens = batch.label_vec[classification_idxs]
        next_tokens = target_tokens

        next_token_scores = classifier_scores.gather(
            -1, next_tokens.unsqueeze(-1)
        ).squeeze(-1)

        classifier_labels = batch.classifier_label[classification_idxs]

        # 1/0 (pos/neg) labels for each next token given the context.
        classifier_labels = classifier_labels.expand_as(next_token_scores).float()

        # Compute BCE loss based on classifier/attribute labels for the next tokens.
        classifier_losses = F.binary_cross_entropy_with_logits(
            next_token_scores,
            classifier_labels,
            reduction='none',
        )

        notnull = target_tokens.ne(self.NULL_IDX)
        classifier_losses *= notnull

        num_target_tokens = notnull.long().sum(dim=-1)

        non_target_indices = torch.ones_like(classifier_scores, dtype=torch.bool)
        non_target_indices.scatter_(-1, next_tokens.unsqueeze(-1), False)

        normalized_classifier_scores = (
            torch.sigmoid(classifier_scores) - 0.5
        ) * notnull.unsqueeze(dim=-1)

        normalized_non_target_classifier_scores = normalized_classifier_scores[
            non_target_indices
        ].reshape(*classifier_scores.shape[:-1], -1)

        normalized_non_target_classifier_scores_squared = (
            normalized_non_target_classifier_scores**2
        )
        normalized_non_target_classifier_scores_mean = (
            normalized_non_target_classifier_scores.mean(dim=-1)
        )
        normalized_non_target_classifier_var = (
            normalized_non_target_classifier_scores.var(dim=-1)
        )

        (
            normalized_non_target_classifier_scores_mean_reshaped,
            num_target_tokens_reshaped,
        ) = self._reshape_to_record_metrics(
            batch,
            normalized_non_target_classifier_scores_mean.mean(-1),
            num_target_tokens,
            classification_idxs,
        )
        self.record_local_metric(
            'classifier_score_mean',
            AverageMetric.many(
                normalized_non_target_classifier_scores_mean_reshaped,
                num_target_tokens_reshaped,
            ),
        )

        (
            normalized_non_target_classifier_var_reshaped,
            num_target_tokens_reshaped,
        ) = self._reshape_to_record_metrics(
            batch,
            normalized_non_target_classifier_var.mean(-1),
            num_target_tokens,
            classification_idxs,
        )
        self.record_local_metric(
            'classifier_score_var',
            AverageMetric.many(
                normalized_non_target_classifier_var_reshaped,
                num_target_tokens_reshaped,
            ),
        )

        # Explicitly force the score for non-target tokens to 0.5.
        # This is done as << 0.5 indicates negative attributes and
        # >> 0.5 indicates positive attributes.
        if self.explicit_classifier_norm:
            classifier_losses += (
                self.explicit_classifier_norm_coeff
                * normalized_non_target_classifier_scores_squared.mean(dim=-1)
            )

        classifier_losses = classifier_losses.sum(dim=1)

        (
            classifier_losses_reshaped,
            num_target_tokens_reshaped,
        ) = self._reshape_to_record_metrics(
            batch, classifier_losses, num_target_tokens, classification_idxs
        )
        self.record_local_metric(
            'classifier_loss',
            AverageMetric.many(classifier_losses_reshaped, num_target_tokens_reshaped),
        )

        classifier_predictions = (torch.sigmoid(next_token_scores) > 0.5).long()

        classifier_accuracy = classifier_labels == classifier_predictions
        classifier_accuracy *= notnull
        classifier_accuracy = classifier_accuracy.sum(-1)
        (
            classifier_accuracy_reshaped,
            num_target_tokens_reshaped,
        ) = self._reshape_to_record_metrics(
            batch, classifier_accuracy, num_target_tokens, classification_idxs
        )
        self.record_local_metric(
            'classifier_accuracy',
            AverageMetric.many(
                classifier_accuracy_reshaped, num_target_tokens_reshaped
            ),
        )

        f1s = {}
        for class_name, positive_class in (('neg', 0), ('pos', 1)):
            positives = classifier_labels == positive_class
            negatives = classifier_labels != positive_class
            trues = classifier_predictions == classifier_labels
            falses = classifier_predictions != classifier_labels

            true_positives = ((positives & trues) * notnull).sum(-1)
            false_positives = ((negatives & falses) * notnull).sum(-1)
            false_negatives = ((positives & falses) * notnull).sum(-1)

            classifier_f1 = (2 * true_positives) / (
                2 * true_positives + false_positives + false_negatives
            )
            classifier_f1[true_positives == 0] = 0

            (classifier_f1_reshaped, _) = self._reshape_to_record_metrics(
                batch, classifier_f1, num_target_tokens, classification_idxs
            )

            f1s[class_name] = classifier_f1_reshaped

            batch_positives = batch.classifier_label[:, 0] == positive_class
            # We use (classification_idxs & (batch_positives > 0) to indicate that we only consider the exs
            # that are ltr classification examples and classifier_labels == positive_class.
            self.record_local_metric(
                f'{class_name}_classifier_f1',
                AverageMetric.many(
                    classifier_f1_reshaped,
                    (classification_idxs & (batch_positives > 0)).int(),
                ),
            )

        avg_classifier_f1_reshaped = sum(f1s.values())
        # We use classification_idxs.int()  to indicate that we only consider the exs that are
        # ltr classification examples.
        self.record_local_metric(
            f'classifier_f1',
            AverageMetric.many(avg_classifier_f1_reshaped, classification_idxs.int()),
        )
        return classifier_losses_reshaped, num_target_tokens_reshaped

    def compute_generator_loss(self, generator_scores, batch):
        bsz = batch.batchsize
        device = generator_scores.device

        generator_losses = torch.zeros((bsz,), device=device)
        num_target_tokens = torch.zeros((bsz,), device=device, dtype=torch.long)

        generation_idxs = torch.logical_not(batch.is_ltr[:, 0])
        self.record_local_metric(
            'pct_generator_exs', AverageMetric.many(generation_idxs)
        )

        # If there are no generation exs in the batch,
        # returrn zero generator loss.
        if not torch.any(generation_idxs):
            return generator_losses, num_target_tokens

        # Copied verbatim from TGA.compute_loss.
        generator_scores = generator_scores[generation_idxs]
        generator_label_vec = batch.label_vec[generation_idxs]

        generator_scores_view = generator_scores.reshape(-1, generator_scores.size(-1))
        generator_losses = self.criterion(
            generator_scores_view, generator_label_vec.view(-1)
        )

        # cross entropy loss
        generator_losses = generator_losses.view(generator_scores.shape[:-1]).sum(dim=1)

        notnull = generator_label_vec.ne(self.NULL_IDX)
        num_target_tokens = notnull.long().sum(dim=-1)

        (
            reshaped_generator_losses,
            num_target_tokens_reshaped,
        ) = self._reshape_to_record_metrics(
            batch, generator_losses, num_target_tokens, generation_idxs
        )

        # save loss to metrics
        self.record_local_metric(
            'generator_loss',
            AverageMetric.many(reshaped_generator_losses, num_target_tokens_reshaped),
        )

        #  save perplexity to metrics
        self.record_local_metric(
            'generator_ppl',
            PPLMetric.many(reshaped_generator_losses, num_target_tokens_reshaped),
        )

        return reshaped_generator_losses, num_target_tokens_reshaped

    def compute_loss(self, batch, return_output=False):
        """
        Overrides compute_loss for multi-objective Director loss computation.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        generator_scores, _, classifier_scores, *_ = model_output

        generator_losses, generator_num_target_tokens = self.compute_generator_loss(
            generator_scores, batch
        )
        classifier_losses, classifier_num_target_tokens = self.compute_classifier_loss(
            classifier_scores, batch
        )

        losses = generator_losses + self.train_gamma * classifier_losses

        num_target_tokens = generator_num_target_tokens + classifier_num_target_tokens

        self.record_local_metric('loss', AverageMetric.many(losses, num_target_tokens))

        # This unweighted_loss ignores mixing weights and weighs
        # generator and classifier losses equally. This can be
        # used to do validation across various mixing coeffs (train_gamma).
        self.record_local_metric(
            'unweighted_loss',
            AverageMetric.many(
                (generator_losses + classifier_losses), num_target_tokens
            ),
        )

        loss = (losses / num_target_tokens).sum()

        if return_output:
            return (loss, model_output)
        return loss
