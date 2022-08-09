#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Dict, Tuple, Union, List
from abc import abstractmethod
from parlai.agents.transformer.modules import (
    TransformerDecoderLayer,
)
from parlai.agents.fid.fid import (
    FidAgent,
    FidModel,
)
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import PPLMetric
from parlai.core.torch_agent import Batch, DictionaryAgent
import parlai.utils.logging as logging
from projects.blenderbot2.agents.blenderbot2 import (
    BlenderBot2FidAgent,
    BlenderBot2FidModel,
    T5BlenderBot2FidModel,
)
from parlai.agents.fid.fid import (
    WizIntGoldDocRetrieverFiDAgent,
)
import parlai.agents.transformer.modules.ffn as ffn
from parlai.agents.transformer.modules import (
    LAYER_NORM_EPS,
)


class DirectorFidModelMixin:
    def __init__(self, opt: Opt, dictionary: DictionaryAgent, **kwargs):
        super().__init__(opt, dictionary, **kwargs)

        vocabulary_size = len(dictionary)

        decoder_output_dim = self.seq2seq_decoder.out_dim
        self.classifier_head = nn.Linear(decoder_output_dim, vocabulary_size)

        # TODO which one is train_gamma, infer_gamma
        self.train_mixing_weights = opt['train_mixing_weights']
        self.infer_mixing_weights = self.train_mixing_weights
        if opt.get('infer_mixing_weights') is not None:
            self.infer_mixing_weights = opt.get('infer_mixing_weights')

        self.use_one_plus_gamma_variant = opt['use_one_plus_gamma_variant']
        self.infer_gamma = opt['train_gamma']
        if opt.get('infer_gamma') is not None:
            self.infer_gamma = opt['infer_gamma']

        self.freeze_decoder = opt['freeze_decoder']

        self.classifier_layer_choice = opt.get('director_classifier_layers', 'linear')
        if self.classifier_layer_choice == 'ffn':
            self.classifier_ffn = ffn.TransformerFFN(
                opt=opt,
                dim=opt['embedding_size'],
                dim_hidden=opt['ffn_size'],
                relu_dropout=opt['relu_dropout'],
                activation='relu',
            )
            self.classifier_norm3 = torch.nn.LayerNorm(
                opt['embedding_size'], eps=LAYER_NORM_EPS
            )
            self.classifier_variant = self.seq2seq_decoder.variant
            self.classifier_dropout = self.seq2seq_decoder.dropout
        elif self.classifier_layer_choice == 'layer':
            self.classifier_layer = TransformerDecoderLayer(
                opt,
                n_heads=opt['n_heads'],
                embedding_size=opt['embedding_size'],
                ffn_size=opt['ffn_size'],
                attention_dropout=opt['attention_dropout'],
                relu_dropout=opt['relu_dropout'],
                dropout=opt['dropout'],
            )
        elif self.classifier_layer_choice == '2layers':
            self.classifier_layer1 = TransformerDecoderLayer(
                opt,
                n_heads=opt['n_heads'],
                embedding_size=opt['embedding_size'],
                ffn_size=opt['ffn_size'],
                attention_dropout=opt['attention_dropout'],
                relu_dropout=opt['relu_dropout'],
                dropout=opt['dropout'],
            )
            self.classifier_layer2 = TransformerDecoderLayer(
                opt,
                n_heads=opt['n_heads'],
                embedding_size=opt['embedding_size'],
                ffn_size=opt['ffn_size'],
                attention_dropout=opt['attention_dropout'],
                relu_dropout=opt['relu_dropout'],
                dropout=opt['dropout'],
            )

    def generator_output(self, input: torch.Tensor):
        if self.freeze_decoder:
            input = input.detach()

        return super().decoder_output(input)

    def classifier_output(
        self,
        input: torch.Tensor,
        encoder_states: Optional[Tuple[Any, ...]] = None,
    ):
        if self.freeze_decoder:
            input = input.detach()

        classifier_outputs = None
        if self.classifier_layer_choice == 'linear':
            classifier_outputs = self.classifier_head(input)
        elif self.classifier_layer_choice == 'ffn':
            ffn_output = input
            residual = input
            if self.classifier_variant == 'prelayernorm':
                ffn_output = self.classifier_norm3(ffn_output)
            ffn_output = self.classifier_ffn(ffn_output)
            ffn_output = self.classifier_dropout(ffn_output)  # --dropout
            ffn_output = residual + ffn_output
            if (
                self.classifier_variant == 'aiayn'
                or self.classifier_variant == 'xlm'
                or self.classifier_variant == 'bart'
            ):
                ffn_output = self.classifier_norm3(ffn_output)
            classifier_outputs = self.classifier_head(ffn_output)
        elif self.classifier_layer_choice == 'layer':
            encoder_output = None
            encoder_mask = None
            if encoder_states is not None:
                encoder_output, encoder_mask, *_ = encoder_states
            layer_out = self.classifier_layer(
                x=input,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
            )
            classifier_outputs = self.classifier_head(layer_out[0])
        elif self.classifier_layer_choice == '2layers':
            encoder_output = None
            encoder_mask = None
            if encoder_states is not None:
                encoder_output, encoder_mask, *_ = encoder_states
            layer_out1 = self.classifier_layer1(
                x=input,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
            )
            layer_out2 = self.classifier_layer2(
                x=layer_out1[0],
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
            )
            classifier_outputs = self.classifier_head(layer_out2[0])
        return classifier_outputs

    def decoder_output(
        self, latent: torch.Tensor, encoder_states: Optional[Tuple[Any, ...]] = None
    ):
        """
        Overriding decoder_output method to use classifier heads to modify the generator
        logprobs. This modification allows model to incorporate attribute information
        from classifier while selecting the next tokens.

        Notice that the output method of a RagModel is an identity mapping. One need to override the RagModel.decoder_output instead

        Args:
            latent (torch.Tensor): decoder outputs
            encoder_states (tuple[Any, ...]): encoder states after the final decoder layer of seq2seq_decoder

        Returns:
            Modified logprobs.
        """
        classifier_outputs = F.logsigmoid(
            self.classifier_output(latent, encoder_states=encoder_states)
        )
        log_predictor_scores = F.log_softmax(self.generator_output(latent), dim=-1)

        if self.use_one_plus_gamma_variant:
            scores = log_predictor_scores + self.infer_gamma * classifier_outputs
        else:
            scores = (
                2.0 * (1.0 - self.infer_mixing_weights) * log_predictor_scores
                + 2.0 * (self.infer_mixing_weights) * classifier_outputs
            )
        return F.log_softmax(scores, dim=-1)

    def load_state_dict(self, state_dict):
        """
        Overrided to load only the generator weights from the state dict and leaving the
        classifier head weights untouched.
        """
        overriden_keys = []
        for k, v in self.state_dict().items():
            if k not in state_dict:
                overriden_keys.append(k)
                state_dict[k] = v
        if overriden_keys:
            logging.debug(f"Those state keys are override: {overriden_keys}")

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

        classifer_score = self.classifier_output(
            input=latent, encoder_states=encoder_states
        )
        return scores, preds, classifer_score, latent, mask, encoder_states

    def decode_forced(
        self, encoder_states: Tuple[torch.Tensor, ...], ys: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.BoolTensor]:
        """
        Override RAG.decode_forced to return latent states and using generator_output
        method to generate the decoder output.
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        if (ys[:, 0] == self.START_IDX).any() and self.generation_model != 'bart':
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        doc_scores = encoder_states[4]

        inputs = self._rag_model_interface.get_initial_forced_decoder_input(
            bsz,
            inputs,
            n_docs=doc_scores.size(1) if doc_scores is not None else None,
            start_idx=self.START_IDX,
            end_idx=self.END_IDX,
            input_turns_cnt=encoder_states[2],
        )
        latent, mask = self.decoder_output_before_final_projection(
            inputs, encoder_states
        )
        logits = self.generator_output(latent)
        _, preds = logits.max(dim=-1)
        return logits, preds, latent, mask

    def decoder_output_before_final_projection(
        self,
        input: torch.LongTensor,
        encoder_states: Tuple[Any, ...],
        incr_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        output the seq2seq decoder output before applying final_output.
        """
        enc_out, enc_mask, *_ = encoder_states
        dec_out, incr_state = self.seq2seq_decoder(
            input, (enc_out, enc_mask), incr_state
        )  # type: ignore
        return dec_out, incr_state

    def decoder(
        self,
        input: torch.LongTensor,
        encoder_states: Tuple[Any, ...],
        incr_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Refactored Decode, RAG-Style to split the FID.decoder into
        decoder_output_before_final_projection + decoder_output.

        :param input:
            input for the decoder
        :param encoder_states:
            RAG encoder states
        :param incr_state:
            incremental decoder state

        :return (output, new_incr_state):
            return the output token distribution, as well as new incremental state.
        """

        dec_out, incr_state = self.decoder_output_before_final_projection(
            input, encoder_states
        )
        dec_out = self.decoder_output(dec_out, encoder_states)
        return dec_out, incr_state


class DirectorBlenderBot2FidModel(DirectorFidModelMixin, BlenderBot2FidModel):
    pass


class DirectorFidAgent(FidAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        # This method will add arguments specific to fused classifier architecture
        # like num_layers classifier, etc.
        super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group('DirectorFidAgent Group')
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
            help='norm coefficient if we should explictly try to set non-target tokens to 0.5 during training.',
        )
        group.add_argument(
            '--freeze-decoder',
            type=bool,
            default=False,
            help='Freeze decoder for training only classifier head.',
        )
        group.add_argument(
            '--use-one-plus-gamma-variant',
            type=bool,
            default=False,
            help='Use 1+ gamma weighting variant if True (keeping generator loss weight fixed).',
        )
        group.add_argument(
            '--train-gamma',
            type=float,
            default=0.5,
            help="Keeping generator loss weight fixed (to 1) and using gamma to weight classifier during training time.",
        )
        group.add_argument(
            '--infer-gamma',
            type=float,
            default=None,
            help="Keeping generator loss weight fixed (to 1) and using gamma to weight classifier during inference time.",
        )
        group.add_argument(
            '--train-mixing-weights',
            type=float,
            default=0.5,
            help='Mixing weights between classifier loss and generator loss during training time',
        )
        group.add_argument(
            '--infer-mixing-weights',
            type=float,
            default=None,
            help='Mixing weights between classifier loss and generator loss during inference time',
        )
        parser.add_argument(
            '--pos-class-weight',
            type=float,
            default=1,
            help='weight of each of the positive class for the sigmoid',
        )
        group.add_argument(
            '--director-classifier-layers',
            type=str,
            default='linear',
            choices=['linear', 'layer', 'ffn', '2layers'],
        )
        group.add_argument(
            '--init-classifier-layers',
            type=str,
            default='random',
            choices=['random', 'copy_decoder', 'copy_decoder_with_epsilon_noise'],
            help="how to init weights of the classifier_layers, such as init randomly, or copy from pretrained decoder, or copy with epsilon pertubation",
        )
        group.add_argument(
            '--init-epsilon',
            type=float,
            default=0,
        )
        return parser

    def __init__(self, opt: Opt, shared=None):
        self.init_classifier_layers = opt['init_classifier_layers']
        self.init_epsilon = opt['init_epsilon']
        if opt['init_classifier_layers'] != 'copy_decoder_with_epsilon_noise':
            self.init_epsilon = 0
        super().__init__(opt, shared)
        self.explicit_classifier_norm = opt['explicit_classifier_norm']
        self.explicit_classifier_norm_coeff = opt['explicit_classifier_norm_coeff']
        self.train_mixing_weights = opt['train_mixing_weights']
        self.use_one_plus_gamma_variant = opt['use_one_plus_gamma_variant']
        self.train_gamma = opt['train_gamma']
        self.infer_gamma = opt['infer_gamma']
        self.pos_class_weight = opt['pos_class_weight']

        assert opt[
            'beam_block_full_context'
        ], 'must set --beam-block-full-context True to use PACER'
        assert (
            opt['rag_model_type'] == 'token'
        ), "compute_loss not supporting other rag model type"

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This copies the classifier specific params to init_model and then calls the
        load_state_dict method of TorchAgent.
        """

        # the random_model_states: model states to be overriden by init model states
        random_model_states = self.model_api.state_dict()
        if 'copy_decoder' in self.init_classifier_layers:
            new_states_to_init = [k for k in random_model_states if k not in state_dict]
            n_decoder_layers = self.model_api.seq2seq_decoder.n_layers
            key_prefix_mapping = {
                'classifier_ffn.': f'seq2seq_decoder.layers.{n_decoder_layers-1}.ffn.',
                'classifier_norm3.': f'seq2seq_decoder.layers.{n_decoder_layers-1}.norm3.',
                'classifier_layer.': f'seq2seq_decoder.layers.{n_decoder_layers-1}.',
                'classifier_layer1.': f'seq2seq_decoder.layers.{n_decoder_layers-2}.',
                'classifier_layer2.': f'seq2seq_decoder.layers.{n_decoder_layers-1}.',
            }
            for model_prefix, decoder_prefix in key_prefix_mapping.items():
                states_to_update = [
                    k for k in new_states_to_init if k.startswith(model_prefix)
                ]
                for new_key in states_to_update:
                    decoder_key = decoder_prefix + new_key[len(model_prefix) :]
                    state_dict[new_key] = (
                        random_model_states[new_key].cpu() * self.init_epsilon
                        + state_dict[decoder_key]
                    )

        # classifier head states or all classifier_layers states if not copying from decoder
        for k, v in random_model_states.items():
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

    @abstractmethod
    def build_model(self) -> FidModel:
        """
        build fid models.
        """

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

    def _reshape_tensor(self, new_len, tensor, indices):
        """
        This method reshapes the tensor back to the batch size.

        Args:
            batch: batch being processed in this iteration.
            tensor: vector (shape: b' X 1), where b' <= b.
            indices: indices of (either classification or generation) exs for which the loss was computed.

        Returns:
            reshaped tensor of shape: b X 1.
        """
        reshaped_tensor = torch.zeros(new_len, device=tensor.device, dtype=tensor.dtype)
        reshaped_tensor[indices] = tensor
        return reshaped_tensor

    def _reshape_to_record_metrics(self, batch, losses, num_target_tokens, indices):
        """
        MultitaskAgent shuffles and combines examples from both classifier and the
        generator tasks in a single batch. We compute losses only for those exs in the
        batch resulting in losses and num_target_tokens vectors that are smaller than
        the.

        This method reshapes the losses and num_target_tokens vectors back to the batch size.
        This is needed to record local metrics as the metrics need to be of batch size for classifiers.

        Args:
            batch: batch being processed in this iteration.
            losses: classifier or generator loss vector (shape: b' X 1), where b' <= b.
            num_target_tokens: number of tokens in each examples for classification or generation tasks. (shape: b' X 1), where b' <= b.
            indices: indices of (either classification or generation) exs for which the loss was computed.

        Returns:
            A tuple of reshaped losses and num_target_tokens, both of shape: b X 1.
        """
        val_id_shape = batch.valid_indices.shape
        reshaped_losses = self._reshape_tensor(val_id_shape, losses, indices)
        reshaped_num_target_tokens = self._reshape_tensor(
            val_id_shape, num_target_tokens, indices
        )

        return (reshaped_losses, reshaped_num_target_tokens)

    def _reshape_generator_tensor_to_record_metrics(
        self, batch, losses, metric_loss, metric_correct, metric_target_tokens, indices
    ):
        """
        This method reshapes the loss-related vectors back to the batch size. This is
        needed to record local metrics as the metrics need to be of batch size for
        generator model.

        Args:
            batch: batch being processed in this iteration.
            losses: losses we use for back propogate (shape: b' X 1), where b' <= b.
            metric_loss: loss we use for metrics (shape: b' X 1), where b' <= b.
            metric_correct: correct predictions from the model (shape: b' X 1), where b' <= b.
            metric_target_tokens: number of tokens in each examples for classification or generation tasks. (shape: b' X 1), where b' <= b.
            indices: indices of (either classification or generation) exs for which the loss was computed.

        Returns:
            A tuple of reshaped losses, metric_loss, metric_correct and num_target_tokens, all of shape: b X 1.
        """
        val_id_shape = batch.valid_indices.shape

        reshaped_losses = self._reshape_tensor(val_id_shape, losses, indices)
        reshaped_metric_loss = self._reshape_tensor(val_id_shape, metric_loss, indices)
        reshaped_metric_correct = self._reshape_tensor(
            val_id_shape, metric_correct, indices
        )
        reshaped_num_target_tokens = self._reshape_tensor(
            val_id_shape, metric_target_tokens, indices
        )

        return (
            reshaped_losses,
            reshaped_metric_loss,
            reshaped_metric_correct,
            reshaped_num_target_tokens,
        )

    def compute_classifier_loss(self, classifier_scores, batch):
        if classifier_scores.size(1) != batch.label_vec.size(1):
            assert self.generation_model == 'bart'
            # ignore start
            classifier_scores = classifier_scores[:, 1:, :]
        bsz = batch.batchsize
        device = classifier_scores.device

        classifier_losses = torch.zeros((bsz,), device=device)
        num_target_tokens = torch.zeros((bsz,), device=device, dtype=torch.long)

        # idxs of all the classification exs in the batch.
        classification_idxs = batch.is_ltr[:, 0]

        # log %classifier examples in the batch
        self.record_local_metric(
            'pct_classifier_exs', AverageMetric.many(classification_idxs)
        )

        # if none of the exs are classifier examples,
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
        # Exapand classifier weights to shape classifier_labels and
        # set them to zero for neg examples.
        classifier_labels = classifier_labels.expand_as(next_token_scores).float()

        # Compute BCE loss based on classifier/attribute labels for the next tokens.
        classifier_losses = F.binary_cross_entropy_with_logits(
            next_token_scores,
            classifier_labels,
            reduction='none',
            pos_weight=self.pos_class_weight * classifier_labels.float(),
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
            'director_classifier_score_mean',
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
            'director_classifier_score_var',
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
            'director_classifier_loss',
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
            'director_classifier_accuracy',
            AverageMetric.many(
                classifier_accuracy_reshaped, num_target_tokens_reshaped
            ),
        )

        # Compute F1 Score.
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
                f'director_{class_name}_classifier_f1',
                AverageMetric.many(
                    classifier_f1_reshaped,
                    (classification_idxs & (batch_positives > 0)).int(),
                ),
            )

        avg_classifier_f1_reshaped = sum(f1s.values())
        # We use classification_idxs.int()  to indicate that we only consider the exs that are

        # ltr classification examples.
        self.record_local_metric(
            'director_classifier_f1',
            AverageMetric.many(avg_classifier_f1_reshaped, classification_idxs.int()),
        )
        return classifier_losses_reshaped, num_target_tokens_reshaped

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
            import ipdb

            ipdb.set_trace()
        return txt

    def get_model_output(self, batch: Batch) -> Tuple[Any, ...]:
        """
        Override RagAgent.get_model_output model output.

        :param batch:
            batch to process

        :return model_output:
            return output from model
        """
        if not self.regret:
            model_output = self.model(
                *self._model_input(batch), ys=batch.label_vec
            )  # type: ignore
            scores, preds, classifier_scores, _, _, enc_state = model_output
        else:
            with torch.no_grad():
                beam_preds_scores, beams = self._regret_generate(
                    batch, self.beam_size, self.regret_intermediate_maxlen
                )
            regret_preds, _, _ = zip(*beam_preds_scores)
            new_batch = self._regret_rebatchify(batch, regret_preds)  # type: ignore
            regret_model_output = self.model(
                *self._model_input(new_batch), ys=batch.label_vec
            )  # type: ignore
            (
                regret_scores,
                preds,
                classifier_scores,
                _,
                _,
                enc_state,
            ) = regret_model_output
            scores = regret_scores

        return (scores, preds, classifier_scores, enc_state)

    def compute_rag_token_loss(
        self,
        criterion: torch.nn.Module,
        scores: torch.Tensor,
        preds: torch.LongTensor,
        enc_state: Tuple[Any],
        label_vec: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Compute RAG Token Loss. Reimplement to get the loss tensor (reduction = 'none')

        This is a simple NLL Loss.

        :param criterion:
            presumably the NLL criterion.
        :param scores:
            model scores
        :param preds:
            model "predicions" of tokens
        :param enc_state:
            encoder states
        :param label_vec:
            target tokens

        :return (losses, metric_loss, correct_tokens, target_tokens):
            loss: the loss through which we backpropagate
            metric_loss: loss we use for metrics
            correct_tokens: correct predictions from the model
            target_tokens: the ground truth tokens.
        """
        if scores.size(1) != label_vec.size(1):
            assert self._rag_model_interface.generation_model == 'bart'
            # ignore start
            scores = scores[:, 1:, :]
            preds = preds[:, 1:]  # type: ignore

        # compute loss
        score_view = scores.reshape(-1, scores.size(-1))
        losses = criterion(score_view, label_vec.view(-1))
        losses = losses.view(scores.shape[:-1]).sum(dim=1)

        # calculate metric counters
        metric_loss = losses.tolist()
        notnull = label_vec.ne(self._rag_model_interface.null_idx)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((label_vec == preds) * notnull).sum(dim=-1)

        # loss = losses.sum()
        # loss /= target_tokens.sum()  # average loss per token
        return losses, metric_loss, correct, target_tokens

    def _get_batch_generator_attr(self, batch, generation_idxs, attr_name=None):
        attr = batch
        if attr_name:
            attr = getattr(batch, attr_name)
        if attr is None:
            return attr
        if torch.is_tensor(attr):
            return attr[generation_idxs]
        else:
            if isinstance(attr, List):
                return [
                    attr[i]
                    for i, i_is_generation in enumerate(generation_idxs)
                    if i_is_generation
                ]
            else:
                raise RuntimeError('unknown type for _get_batch_generator_attr')

    def _reshape_and_record_retrieval_metrics(self, batch, enc_state, generation_idxs):
        generator_batch = Batch(
            batchsize=len(batch.valid_indices[generation_idxs]),
            is_training=batch.is_training,
            text_vec=self._get_batch_generator_attr(batch, generation_idxs, 'text_vec'),
            label_vec=self._get_batch_generator_attr(
                batch, generation_idxs, 'label_vec'
            ),
            labels=self._get_batch_generator_attr(batch, generation_idxs, 'labels'),
            valid_indices=self._get_batch_generator_attr(
                batch, generation_idxs, 'valid_indices'
            ),
            candidates=self._get_batch_generator_attr(
                batch, generation_idxs, 'candidates'
            ),
            candidate_vecs=self._get_batch_generator_attr(
                batch, generation_idxs, 'candidate_vecs'
            ),
            image=self._get_batch_generator_attr(batch, generation_idxs, 'image'),
            rewards=self._get_batch_generator_attr(batch, generation_idxs, 'rewards'),
            observations=self._get_batch_generator_attr(
                batch, generation_idxs, 'observations'
            )
            if self.is_debug
            else None,
            _context_original_length=self._get_batch_generator_attr(
                batch, generation_idxs, '_context_original_length'
            ),
            _context_truncate_rate=self._get_batch_generator_attr(
                batch, generation_idxs, '_context_truncate_rate'
            ),
            _context_truncated_length=self._get_batch_generator_attr(
                batch, generation_idxs, '_context_truncated_length'
            ),
            _label_original_length=self._get_batch_generator_attr(
                batch, generation_idxs, '_label_original_length'
            ),
            _label_truncate_rate=self._get_batch_generator_attr(
                batch, generation_idxs, '_label_truncate_rate'
            ),
            _label_truncated_length=self._get_batch_generator_attr(
                batch, generation_idxs, '_label_truncated_length'
            ),
        )

        generator_enc_state = (
            self._get_batch_generator_attr(enc_state[0], generation_idxs),
            self._get_batch_generator_attr(enc_state[1], generation_idxs),
            self._get_batch_generator_attr(enc_state[2], generation_idxs),
            self._get_batch_generator_attr(enc_state[3], generation_idxs),
            self._get_batch_generator_attr(enc_state[4], generation_idxs),
        )

        self._record_retrieval_metrics(generator_batch, generator_enc_state)

    def compute_generator_loss(self, generator_scores, batch, preds, enc_state):
        bsz = batch.batchsize
        device = generator_scores.device

        generator_losses = torch.zeros((bsz,), device=device)
        num_target_tokens = torch.zeros((bsz,), device=device, dtype=torch.long)

        generation_idxs = torch.logical_not(batch.is_ltr[:, 0])
        self.record_local_metric(
            'pct_generator_exs', AverageMetric.many(generation_idxs)
        )

        # If there are no generation exs in the batch, returrn
        # zero generator loss.
        if not torch.any(generation_idxs):
            return generator_losses, num_target_tokens

        self._reshape_and_record_retrieval_metrics(batch, enc_state, generation_idxs)

        # Copied verbatim from RAG.compute_loss.
        # TODO: record local metrics from BlenderBot2.compute_loss
        generator_scores = generator_scores[generation_idxs]
        generator_label_vec = batch.label_vec[generation_idxs]
        generator_preds = preds[generation_idxs]

        generator_enc_state = (
            self._get_batch_generator_attr(enc_state[0], generation_idxs),
            self._get_batch_generator_attr(enc_state[1], generation_idxs),
            self._get_batch_generator_attr(enc_state[2], generation_idxs),
            self._get_batch_generator_attr(enc_state[3], generation_idxs),
            self._get_batch_generator_attr(enc_state[4], generation_idxs),
        )

        (
            generator_losses,
            generator_metric_loss,
            generator_metric_correct,
            generator_metric_target_tokens,
        ) = self.compute_rag_token_loss(
            self.criterion,
            generator_scores,
            generator_preds,
            generator_enc_state,
            generator_label_vec,
        )

        # save loss to metrics
        if not torch.is_tensor(generator_metric_loss):
            generator_metric_loss = torch.tensor(generator_metric_loss).to(
                generator_scores
            )

        (
            reshaped_generator_losses,
            reshaped_generator_metric_loss,
            reshaped_generator_metric_correct,
            num_target_tokens_reshaped,
        ) = self._reshape_generator_tensor_to_record_metrics(
            batch,
            generator_losses,
            generator_metric_loss,
            generator_metric_correct,
            generator_metric_target_tokens,
            generation_idxs,
        )

        # cross entropy loss
        self.record_local_metric(
            'director_generator_loss',
            AverageMetric.many(
                reshaped_generator_metric_loss, num_target_tokens_reshaped
            ),
        )

        # perplexity
        self.record_local_metric(
            'director_generator_ppl',
            PPLMetric.many(reshaped_generator_metric_loss, num_target_tokens_reshaped),
        )

        self.record_local_metric(
            'director_generator_token_acc',
            AverageMetric.many(
                reshaped_generator_metric_correct, num_target_tokens_reshaped
            ),
        )
        self.record_local_metric(
            'director_generator_token_em',
            AverageMetric.many(
                [
                    x == y
                    for x, y in zip(
                        reshaped_generator_metric_correct, num_target_tokens_reshaped
                    )
                ]
            ),
        )

        return reshaped_generator_losses, num_target_tokens_reshaped

    def compute_loss(self, batch, return_output=False):
        """
        Override compute_loss for multi-objective loss computation.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        model_output = self.get_model_output(batch)
        generator_scores, preds, classifier_scores, enc_state = model_output

        generator_losses, generator_num_target_tokens = self.compute_generator_loss(
            generator_scores, batch, preds, enc_state
        )
        classifier_losses, classifier_num_target_tokens = self.compute_classifier_loss(
            classifier_scores,
            batch,
        )
        assert generator_losses.size(0) == classifier_losses.size(
            0
        ), 'unmatched generator and classifier sizes'
        if self.use_one_plus_gamma_variant:
            losses = generator_losses + self.train_gamma * classifier_losses
        else:
            losses = (
                2.0 * (1.0 - self.train_mixing_weights) * generator_losses
                + 2.0 * (self.train_mixing_weights) * classifier_losses
            )

        num_target_tokens = generator_num_target_tokens + classifier_num_target_tokens

        self.record_local_metric('loss', AverageMetric.many(losses, num_target_tokens))

        # This unweighted_loss ignores mixing weights and weighs
        # generator and classifier losses equally. This can be
        # used to do validation across various mixing coeffs.
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


class DirectorBlenderBot2FidAgent(DirectorFidAgent, BlenderBot2FidAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        BlenderBot2FidAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser

    def build_model(self) -> BlenderBot2FidModel:
        if self.generation_model == 't5':
            model = T5BlenderBot2FidModel(self.opt, self.dict)
        else:
            model = DirectorBlenderBot2FidModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model


"""
parlai train_model -vtim 300 -bs 6 --gradient-clip 10.0 --fp16 True -lr 1e-06 --validation-metric unweighted_loss --validation-metric-mode min --log-every-n-secs 10 -t parlai.tasks.blended_skill_talk.agents:BlendedSkillTalkTeacher:mutators=flatten --model parlai_internal.projects.continual_learning.director_agents.director_bb2:EDCBlenderBot2FidAgent  --validation-cutoff 0 --init-opt /checkpoint/jingxu23/projects/continual_learning/sludge_preset_500k.opt --init-model zoo:blenderbot2/blenderbot2_400M/model --dpr-model-file zoo:hallucination/bart_rag_token/model --query-model bert_from_parlai_rag --variant prelayernorm --text-truncate 128 --truncate 128 --fp16-impl mem_efficient --optimizer adam --history-add-global-end-token end --delimiter '  ' --lr-scheduler-patience 3 --activation gelu --attention-dropout 0.0 --relu-dropout 0.0 --dict-file zoo:blender/reddit_3B/model.dict --log-every-n-steps 10 --max-train-steps 1000 --validation-every-n-steps 50 --max-train-time 169344.0 --save-after-valid True --batchsize 4 --eval-batchsize 4 --update-freq 2 --load-from-checkpoint true --relu-dropout 0.0 -lr 2e-06 --lr-scheduler reduceonplateau --lr-scheduler-patience 3 --model-parallel True --warmup-updates 100 --gradient-clip 0.1 -vmt ppl --skip-generation True -vmm min -vp 10 --rag-model-type token --retriever-debug-index compressed --n-docs 5 --n-extra-positions 0 --max-doc-token-length 64 --truncate 128 --text-truncate 128 --label-truncate 128 --splitted-chunk-length 256 --n-ranked-doc-chunks 1 --search-query-generator-inference beam --search-query-generator-beam-size 1 --search-query-generator-beam-min-length 2 --query-generator-inference beam --query-generator-beam-size 1 --query-generator-beam-min-length 2 --query-generator-truncate -1 --query-generator-ignore-phrase persona: --knowledge-access-method search_only --query-generator-model-file zoo:sea/bart_sq_gen/model --memory-extractor-phrase persona: --memory-key personas --retriever-ignore-phrase "" --memory-decoder-model-file " " --use-one-plus-gamma-variant True --train-gamma 0.01 --model-file /checkpoint/jingxu23/projects/pacer/contradiction_edc/sweeps/train_director_agent_20220502/cb5.job_0/model --skip-generation False --director-classifier-layers layer

parlai em -mf /checkpoint/jingxu23/projects/continual_learning/sweeps/edc_bb2_swp2_20220519/b51.job_29/model -t internal:cl_new_tasks:mutators=flatten_gold_human_mutator_internal,internal:cl_new_tasks:ContinualLearningNewTaskSatisfactionTeacher:mutators=ok_mutator_internal+ok_pos_notok_neg_mutator_internal+cl_EDC_LTR_mutator_internal,internal:cl_new_tasks:ContinualLearningNewTaskSatisfactionTeacher:mutators=notok_mutator_internal+ok_pos_notok_neg_mutator_internal+cl_EDC_LTR_mutator_internal -bs 16 -dt test

parlai train_model -vtim 300 -bs 6 --gradient-clip 10.0 --fp16 True -lr 1e-06 --validation-metric unweighted_loss --validation-metric-mode min --validation-max-exs 1000 --validation-patience 50 --log-every-n-secs 10 --load-from-checkpoint True --save-after-valid True --tensorboard-log True --skip-generation False --aggregate-micro True -t parlai.tasks.blended_skill_talk.agents:BlendedSkillTalkTeacher:mutators=flatten,parlai_internal.projects.pacer.tasks.agents:ContradictionTeacher:mutators=flatten+contradiction_clean_up+EDC_LTR --model parlai_internal.projects.pacer.agents.director_agent:EDCAgent  --validation-cutoff 0 --multitask-weights 1  --embedding-size 1024 --ffn-size 4096 --n-decoder-layers 22 --n-encoder-layers 2 --n-heads 16 --n-positions 2048 --variant prelayernorm --text-truncate 128 --truncate 128 --dict-tokenizer bytelevelbpe --fp16-impl mem_efficient --optimizer adam --update-freq 1 --history-add-global-end-token end --lr-scheduler-patience 3 --warmup-updates 1000 --activation gelu --attention-dropout 0.1 --dropout 0.1 --label-truncate 128 --relu-dropout 0.0 --init-model /checkpoint/parlai/zoo/reddit/400M/model --dict-file /checkpoint/parlai/zoo/reddit/400M/model.dict --use-one-plus-gamma-variant True --train-gamma 0.01 --model-file /checkpoint/jingxu23/projects/pacer/contradiction_edc//sweeps/train_director_agent_20220507/4d6.job_0/model --skip-generation False --validation-every-n-steps 10
"""


class DirectorBlenderBot2WizIntGoldDocRetrieverFiDAgent(
    WizIntGoldDocRetrieverFiDAgent, DirectorBlenderBot2FidAgent
):
    pass
