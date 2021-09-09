#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Code for distilling a transformer/generator model.
"""

from typing import Optional
from parlai.core.params import ParlaiParser
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Union

import torch
from torch import nn as nn
from torch.nn import functional as F

from parlai.agents.bart.bart import BartAgent
from parlai.agents.transformer.modules import (
    MultiHeadAttention,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.agents import create_agent_from_model_file
from parlai.core.metrics import AverageMetric
from parlai.core.opt import Opt
from parlai.core.torch_agent import Batch
from parlai.core.torch_generator_agent import PPLMetric, TorchGeneratorAgent
from parlai.utils.misc import AttrDict
from parlai.utils.torch import NEAR_INF_FP16
from parlai.utils.typing import TShared


class OutputRecorder:
    """
    Saves all outputs from modules that it is registered to.
    """

    def __init__(self):
        self.outputs = []

    def __call__(self, module: nn.Module, module_in: Any, module_out: Any):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class ForwardPassOutputs(AttrDict):
    """
    Provides dot access to all outputs of the forward passes of the encoder and decoder.
    """

    mask: torch.BoolTensor
    decoder_mask: torch.BoolTensor
    tokens_per_example: torch.Tensor
    num_tokens: torch.Tensor
    context_mask: torch.BoolTensor
    context_tokens_per_example: torch.Tensor
    num_context_tokens: torch.Tensor
    task_loss: torch.Tensor
    teacher_scores: torch.Tensor
    teacher_enc_output: torch.Tensor
    teacher_embedding_outputs: Dict[str, torch.Tensor]
    teacher_hidden_states: Dict[str, List[torch.Tensor]]
    teacher_attention_matrices: Dict[str, List[Dict[str, torch.Tensor]]]
    student_output: tuple
    student_scores: torch.Tensor
    student_enc_output: torch.Tensor
    student_embedding_outputs: Dict[str, torch.Tensor]
    student_hidden_states: Dict[str, List[torch.Tensor]]
    student_attention_matrices: Dict[str, List[Dict[str, torch.Tensor]]]

    def __init__(
        self,
        mask,
        decoder_mask,
        tokens_per_example,
        num_tokens,
        context_mask,
        context_tokens_per_example,
        num_context_tokens,
        task_loss,
        teacher_scores,
        teacher_enc_output,
        teacher_embedding_outputs,
        teacher_hidden_states,
        teacher_attention_matrices,
        student_output,
        student_scores,
        student_enc_output,
        student_embedding_outputs,
        student_hidden_states,
        student_attention_matrices,
    ):
        super().__init__(
            mask=mask,
            decoder_mask=decoder_mask,
            tokens_per_example=tokens_per_example,
            num_tokens=num_tokens,
            context_mask=context_mask,
            context_tokens_per_example=context_tokens_per_example,
            num_context_tokens=num_context_tokens,
            task_loss=task_loss,
            teacher_scores=teacher_scores,
            teacher_enc_output=teacher_enc_output,
            teacher_embedding_outputs=teacher_embedding_outputs,
            teacher_hidden_states=teacher_hidden_states,
            teacher_attention_matrices=teacher_attention_matrices,
            student_output=student_output,
            student_scores=student_scores,
            student_enc_output=student_enc_output,
            student_embedding_outputs=student_embedding_outputs,
            student_hidden_states=student_hidden_states,
            student_attention_matrices=student_attention_matrices,
        )


class AbstractDistillTransformerAgentMixin(ABC):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group('AbstractDistillTransformer arguments')
        agent.add_argument('--teacher-model', help='The teacher model file')
        agent.add_argument(
            '--task-loss-coeff',
            type=float,
            default=1,
            help='Coefficient on MLE loss on task',
        )
        agent.add_argument(
            '--encoder-loss-coeff',
            type=float,
            default=0,
            help='Coefficient on teacher loss on encoder output',
        )
        agent.add_argument(
            '--hidden-loss-coeff',
            type=float,
            default=0,
            help='Coefficient on teacher loss on encoder/decoder hidden layers',
        )
        agent.add_argument(
            '--pred-loss-coeff',
            type=float,
            default=0,
            help='Coefficient on KL teacher loss on prediction layer',
        )
        return agent

    def __init__(self, opt, shared=None):

        # Define coefficients
        self.task_loss_coeff = opt['task_loss_coeff']
        self.encoder_loss_coeff = opt['encoder_loss_coeff']
        self.hidden_loss_coeff = opt['hidden_loss_coeff']
        self.pred_loss_coeff = opt['pred_loss_coeff']

        assert (
            opt.get('model_parallel', False) is False
        ), 'model_parallel is not currently supported for distillation!'

        # Create teacher model
        if shared is None:
            to_copy = {'no_cuda', 'model_parallel', 'fp16', 'fp16_impl', 'gpu'}
            override = {k: opt[k] for k in to_copy}
            override['datatype'] = 'train:evalmode'  # Don't initialize the optimizer
            teacher_agent = create_agent_from_model_file(opt['teacher_model'], override)
            self.teacher_agent_opt = teacher_agent.opt
            self.teacher_model = teacher_agent.model
            self.teacher_model.eval()

        super().__init__(opt, shared)

    def build_model(self):

        student_model = super().build_model()

        teacher_model = self._get_teacher_model()

        # Define the mapping between teacher and student encoder layers
        self.student_num_enc_layers = len(student_model.encoder.layers)
        self.teacher_num_enc_layers = len(teacher_model.encoder.layers)
        self.enc_layer_ratio = self.teacher_num_enc_layers / self.student_num_enc_layers
        self.mapped_enc_layers = []
        for i in range(self.student_num_enc_layers):
            j = int((i + 1) * self.enc_layer_ratio) - 1
            self.mapped_enc_layers.append(j)

        # Define the mapping between teacher and student decoder layers
        self.student_num_dec_layers = len(student_model.decoder.layers)
        self.teacher_num_dec_layers = len(teacher_model.decoder.layers)
        self.dec_layer_ratio = self.teacher_num_dec_layers / self.student_num_dec_layers
        self.mapped_dec_layers = []
        for i in range(self.student_num_dec_layers):
            j = int((i + 1) * self.dec_layer_ratio) - 1
            self.mapped_dec_layers.append(j)

        # Register hooks to record outputs
        encoder_module_map = {
            'layers': TransformerEncoderLayer,
            'attentions': MultiHeadAttention,
        }
        decoder_module_map = {
            'layers': TransformerDecoderLayer,
            'attentions': MultiHeadAttention,
        }
        self.hooks = {
            'teacher': {
                'encoder': self._register_series_of_hooks(
                    model=teacher_model.encoder, module_map=encoder_module_map
                ),
                'decoder': self._register_series_of_hooks(
                    model=teacher_model.decoder, module_map=decoder_module_map
                ),
            },
            'student': {
                'encoder': self._register_series_of_hooks(
                    model=student_model.encoder, module_map=encoder_module_map
                ),
                'decoder': self._register_series_of_hooks(
                    model=student_model.decoder, module_map=decoder_module_map
                ),
            },
        }

        # Separately register hooks for the token embeddings, which are the same for
        # the encoder and decoder
        self.hooks['teacher']['embeddings'] = OutputRecorder()
        teacher_model.embeddings.register_forward_hook(
            self.hooks['teacher']['embeddings']
        )
        self.hooks['student']['embeddings'] = OutputRecorder()
        student_model.embeddings.register_forward_hook(
            self.hooks['student']['embeddings']
        )

        return student_model

    def _get_teacher_model(self) -> nn.Module:
        """
        Return the teacher model.

        This logic is needed because the teacher model may be wrapped by
        torch.nn.parallel.DistributedDataParallel.
        """
        if hasattr(self.teacher_model, 'module'):
            return self.teacher_model.module
        else:
            return self.teacher_model

    def _register_series_of_hooks(
        self, model: nn.Module, module_map: Dict[str, Type[nn.Module]]
    ) -> Dict[str, OutputRecorder]:
        """
        Register hooks in modules of the model, given the mapping of module types.

        `module_map` is a dict whose keys are module-type names and whose values are
        module types. For each module type, during each forward pass of `model`, all
        outputs of modules of that type will be saved to `hooks[module_type].outputs`.
        """
        hooks = {}
        for module_name, module_type in module_map.items():
            hooks[module_name] = OutputRecorder()
            for module in model.modules():
                if isinstance(module, module_type):
                    module.register_forward_hook(hooks[module_name])
        return hooks

    @abstractmethod
    def compute_loss(self, batch, return_output=False):
        """
        Return the loss.

        This will likely call self._perform_forward_passes().
        """

    def _perform_forward_passes(self, batch: Batch) -> ForwardPassOutputs:
        """
        Perform forward passes through the student and teacher and pass back outputs.
        """

        assert isinstance(self, TorchGeneratorAgent)
        # Code relies on methods

        mask = batch.label_vec != self.NULL_IDX

        self._clear_hook_outputs(self.hooks)

        # Forward pass through teacher model
        with torch.no_grad():
            teacher_scores, teacher_preds, teacher_enc_states = self.teacher_model(
                *self._model_input(batch), ys=batch.label_vec
            )
            teacher_enc_output, context_mask = teacher_enc_states

        # Forward pass through student model
        task_loss, student_output = super().compute_loss(batch, return_output=True)
        student_scores, student_preds, student_enc_states = student_output
        student_enc_output, _ = student_enc_states

        # Compile all outputs given the hooks
        teacher_embedding_outputs = self._extract_embedding_outputs(
            hooks=self.hooks['teacher']
        )
        student_embedding_outputs = self._extract_embedding_outputs(
            hooks=self.hooks['student']
        )
        teacher_hidden_states = self._extract_hidden_states(
            hooks=self.hooks['teacher'],
            num_enc_layers=self.teacher_num_enc_layers,
            num_dec_layers=self.teacher_num_dec_layers,
        )
        student_hidden_states = self._extract_hidden_states(
            hooks=self.hooks['student'],
            num_enc_layers=self.student_num_enc_layers,
            num_dec_layers=self.student_num_dec_layers,
        )
        teacher_attention_matrices = self._extract_attention_matrices(
            hooks=self.hooks['teacher'],
            num_enc_layers=self.teacher_num_enc_layers,
            num_dec_layers=self.teacher_num_dec_layers,
        )
        student_attention_matrices = self._extract_attention_matrices(
            hooks=self.hooks['student'],
            num_enc_layers=self.student_num_enc_layers,
            num_dec_layers=self.student_num_dec_layers,
        )
        self._clear_hook_outputs(self.hooks)

        tokens_per_example = mask.sum(dim=-1)  # Sum over tokens
        num_tokens = mask.sum()
        context_tokens_per_example = context_mask.sum(dim=-1)  # Sum over tokens
        num_context_tokens = context_mask.sum()

        # If needed, perform further manipulation of the mask tensor
        mask = self._manipulate_mask(
            mask=mask, student_scores=student_scores, batch=batch
        )
        decoder_mask = self._manipulate_mask(
            mask=mask, student_scores=student_embedding_outputs["decoder"], batch=batch
        )

        # Record teacher accuracy
        teacher_acc = ((student_preds == teacher_preds) * mask).sum(dim=-1)
        # Sum over tokens
        self.record_local_metric(
            'teacher_acc', AverageMetric.many(teacher_acc, tokens_per_example)
        )

        return ForwardPassOutputs(
            mask=mask,
            decoder_mask=decoder_mask,
            tokens_per_example=tokens_per_example,
            num_tokens=num_tokens,
            context_mask=context_mask,
            context_tokens_per_example=context_tokens_per_example,
            num_context_tokens=num_context_tokens,
            task_loss=task_loss,
            teacher_scores=teacher_scores,
            teacher_enc_output=teacher_enc_output,
            teacher_embedding_outputs=teacher_embedding_outputs,
            teacher_hidden_states=teacher_hidden_states,
            teacher_attention_matrices=teacher_attention_matrices,
            student_output=student_output,
            student_scores=student_scores,
            student_enc_output=student_enc_output,
            student_embedding_outputs=student_embedding_outputs,
            student_hidden_states=student_hidden_states,
            student_attention_matrices=student_attention_matrices,
        )

    def _manipulate_mask(
        self, mask: torch.BoolTensor, student_scores: torch.Tensor, batch: Batch
    ) -> torch.BoolTensor:
        """
        If necessary, perform further manipulations of the mask.

        Needed for BART-based student models to add in an extra start token.
        """
        if hasattr(super(), '_manipulate_mask'):
            # Defer to any agent-specific method for manipulating the mask
            return super()._manipulate_mask(
                mask=mask, student_scores=student_scores, batch=batch
            )
        else:
            return mask

    def _extract_embedding_outputs(
        self, hooks: Dict[str, Dict[str, OutputRecorder]]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract out the encoder and decoder embedding outputs.
        """
        assert len(hooks['embeddings'].outputs) == 2
        return {
            'encoder': hooks['embeddings'].outputs[0],
            'decoder': hooks['embeddings'].outputs[1],
        }

    def _extract_hidden_states(
        self,
        hooks: Dict[str, Dict[str, OutputRecorder]],
        num_enc_layers: int,
        num_dec_layers: int,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Extract out encoder/decoder hidden states per layer.
        """
        assert len(hooks['encoder']['layers'].outputs) == num_enc_layers
        assert len(hooks['decoder']['layers'].outputs) == num_dec_layers
        return {
            'encoder': hooks['encoder']['layers'].outputs,
            'decoder': [out_[0] for out_ in hooks['decoder']['layers'].outputs],
        }

    def _extract_attention_matrices(
        self,
        hooks: Dict[str, Dict[str, OutputRecorder]],
        num_enc_layers: int,
        num_dec_layers: int,
    ) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """
        Extract out encoder/decoder attention matrices per layer and attention type.
        """
        assert len(hooks['encoder']['attentions'].outputs) == num_enc_layers
        assert len(hooks['decoder']['attentions'].outputs) == 2 * num_dec_layers
        output_idx = 2  # The position of the attention matrix among the outputs
        return {
            'encoder': [
                {
                    'self_attn': hooks['encoder']['attentions'].outputs[layer_idx][
                        output_idx
                    ]
                }
                for layer_idx in range(num_enc_layers)
            ],
            'decoder': [
                {
                    'self_attn': hooks['decoder']['attentions'].outputs[2 * layer_idx][
                        output_idx
                    ],
                    'encoder_attn': hooks['decoder']['attentions'].outputs[
                        2 * layer_idx + 1
                    ][output_idx],
                }
                for layer_idx in range(num_dec_layers)
            ],
        }

    def _clear_hook_outputs(self, hooks: Union[Dict[str, Any], OutputRecorder]):
        """
        Recursively clear outputs from all hooks.
        """
        if isinstance(hooks, dict):
            for subhooks in hooks.values():
                self._clear_hook_outputs(subhooks)
        else:
            # `hooks` is an OutputRecorder
            hooks.clear()

    def _get_encoder_loss(self, fwd_pass: ForwardPassOutputs) -> torch.Tensor:
        """
        Return the loss on the encoder's output layer.
        """
        assert isinstance(self, TorchGeneratorAgent)
        # Code relies on methods
        encoder_loss = F.mse_loss(
            input=fwd_pass.student_enc_output,
            target=fwd_pass.teacher_enc_output,
            reduction='none',
        )
        encoder_loss = encoder_loss.mean(dim=-1) * fwd_pass.context_mask
        # Avg over embedding dim
        self.record_local_metric(
            'enc_loss',
            AverageMetric.many(
                encoder_loss.sum(dim=-1), fwd_pass.context_tokens_per_example
            ),
        )  # Sum over token dim
        encoder_loss = encoder_loss.div(fwd_pass.num_context_tokens).sum()
        # Divide before summing over examples so that values don't get too large
        return encoder_loss

    def _get_embedding_losses(
        self, fwd_pass: ForwardPassOutputs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the encoder and decoder embedding losses.
        """
        assert isinstance(self, TorchGeneratorAgent)
        # Code relies on methods
        enc_emb_loss, enc_emb_loss_per_example = self._get_component_embedding_loss(
            student_emb_output=fwd_pass.student_embedding_outputs['encoder'],
            teacher_emb_output=fwd_pass.teacher_embedding_outputs['encoder'],
            mask=fwd_pass.context_mask,
            num_tokens=fwd_pass.num_context_tokens,
        )
        self.record_local_metric(
            'enc_emb_loss',
            AverageMetric.many(
                enc_emb_loss_per_example, fwd_pass.context_tokens_per_example
            ),
        )
        dec_emb_loss, dec_emb_loss_per_example = self._get_component_embedding_loss(
            student_emb_output=fwd_pass.student_embedding_outputs['decoder'],
            teacher_emb_output=fwd_pass.teacher_embedding_outputs['decoder'],
            mask=fwd_pass.decoder_mask,
            num_tokens=fwd_pass.num_tokens,
        )
        self.record_local_metric(
            'dec_emb_loss',
            AverageMetric.many(dec_emb_loss_per_example, fwd_pass.tokens_per_example),
        )
        return enc_emb_loss, dec_emb_loss

    def _get_component_embedding_loss(
        self,
        student_emb_output: torch.Tensor,
        teacher_emb_output: torch.Tensor,
        mask: torch.BoolTensor,
        num_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the embedding loss for either the encoder or the decoder.
        """
        assert isinstance(self, TorchGeneratorAgent)
        # Code relies on methods
        raw_loss = F.mse_loss(
            input=student_emb_output, target=teacher_emb_output, reduction='none'
        )
        clamped_loss = torch.clamp(raw_loss, min=0, max=NEAR_INF_FP16)
        # Prevent infs from appearing in the loss term. Especially important with fp16
        masked_loss = clamped_loss.mean(dim=-1) * mask
        # Average over embedding dim
        embedding_loss_per_example = masked_loss.sum(dim=-1)  # Sum over token dim
        embedding_loss = masked_loss.div(num_tokens).sum()
        # Divide before summing over examples so that values don't get too large
        return embedding_loss, embedding_loss_per_example

    def _get_hidden_losses(
        self, fwd_pass: ForwardPassOutputs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the encoder and decoder hidden losses.
        """
        assert isinstance(self, TorchGeneratorAgent)
        # Code relies on methods
        enc_hidden_loss, enc_hidden_loss_per_example = self._get_component_hidden_loss(
            student_hidden_states=fwd_pass.student_hidden_states['encoder'],
            teacher_hidden_states=fwd_pass.teacher_hidden_states['encoder'],
            mask=fwd_pass.context_mask,
            num_tokens=fwd_pass.num_context_tokens,
            mapped_layers=self.mapped_enc_layers,
        )
        self.record_local_metric(
            'enc_hid_loss',
            AverageMetric.many(
                enc_hidden_loss_per_example, fwd_pass.context_tokens_per_example
            ),
        )
        dec_hidden_loss, dec_hidden_loss_per_example = self._get_component_hidden_loss(
            student_hidden_states=fwd_pass.student_hidden_states['decoder'],
            teacher_hidden_states=fwd_pass.teacher_hidden_states['decoder'],
            mask=fwd_pass.decoder_mask,
            num_tokens=fwd_pass.num_tokens,
            mapped_layers=self.mapped_dec_layers,
        )
        self.record_local_metric(
            'dec_hid_loss',
            AverageMetric.many(
                dec_hidden_loss_per_example, fwd_pass.tokens_per_example
            ),
        )
        return enc_hidden_loss, dec_hidden_loss

    def _get_component_hidden_loss(
        self,
        student_hidden_states: List[torch.Tensor],
        teacher_hidden_states: List[torch.Tensor],
        mask: torch.BoolTensor,
        num_tokens: torch.Tensor,
        mapped_layers: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss across all hidden layers for either the encoder or the decoder.

        (The loss is averaged across all hidden layers and over the embedding dimension
        so that it doesn't get too high for fp16 tensors.)
        """
        per_layer_losses = []
        per_layer_per_example_losses = []
        for student_layer_idx, teacher_layer_idx in enumerate(mapped_layers):
            raw_layer_loss = F.mse_loss(
                input=student_hidden_states[student_layer_idx],
                target=teacher_hidden_states[teacher_layer_idx],
                reduction='none',
            )
            clamped_layer_loss = torch.clamp(raw_layer_loss, min=0, max=NEAR_INF_FP16)
            # Prevent infs from appearing in the loss term. Especially important with
            # fp16
            masked_layer_loss = clamped_layer_loss.mean(dim=-1) * mask
            # Average over embedding dim
            layer_loss_per_example = masked_layer_loss.sum(dim=-1)  # Sum over token dim
            layer_loss = masked_layer_loss.div(num_tokens).sum()
            # Divide before summing over examples so that values don't get too large
            per_layer_losses.append(layer_loss)
            per_layer_per_example_losses.append(layer_loss_per_example)
        hidden_loss = torch.stack(per_layer_losses).mean()
        hidden_loss_per_example = torch.stack(per_layer_per_example_losses, dim=1).mean(
            dim=1
        )
        return hidden_loss, hidden_loss_per_example

    def _get_attention_losses(
        self, fwd_pass: ForwardPassOutputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return attention losses.

        Compute and return losses on encoder and decoder self-attention and decoder
        enc/dec attention.
        """
        enc_self_attn_loss = self._get_and_record_component_attention_loss(
            student_attention_matrices=fwd_pass.student_attention_matrices['encoder'],
            teacher_attention_matrices=fwd_pass.teacher_attention_matrices['encoder'],
            mask=fwd_pass.context_mask,
            tokens_per_example=fwd_pass.context_tokens_per_example,
            num_tokens=fwd_pass.num_context_tokens,
            mapped_layers=self.mapped_enc_layers,
            attn_type='self_attn',
            metric_name='enc_self_attn_loss',
        )
        dec_self_attn_loss = self._get_and_record_component_attention_loss(
            student_attention_matrices=fwd_pass.student_attention_matrices['decoder'],
            teacher_attention_matrices=fwd_pass.teacher_attention_matrices['decoder'],
            mask=fwd_pass.decoder_mask,
            tokens_per_example=fwd_pass.tokens_per_example,
            num_tokens=fwd_pass.num_tokens,
            mapped_layers=self.mapped_dec_layers,
            attn_type='self_attn',
            metric_name='dec_self_attn_loss',
        )
        enc_dec_attn_loss = self._get_and_record_component_attention_loss(
            student_attention_matrices=fwd_pass.student_attention_matrices['decoder'],
            teacher_attention_matrices=fwd_pass.teacher_attention_matrices['decoder'],
            mask=fwd_pass.decoder_mask,
            tokens_per_example=fwd_pass.tokens_per_example,
            num_tokens=fwd_pass.num_tokens,
            mapped_layers=self.mapped_dec_layers,
            attn_type='encoder_attn',
            metric_name='enc_dec_attn_loss',
        )
        return enc_self_attn_loss, dec_self_attn_loss, enc_dec_attn_loss

    def _get_and_record_component_attention_loss(
        self,
        teacher_attention_matrices: List[Dict[str, torch.Tensor]],
        student_attention_matrices: List[Dict[str, torch.Tensor]],
        mask: torch.BoolTensor,
        tokens_per_example: torch.Tensor,
        num_tokens: torch.Tensor,
        mapped_layers: List[int],
        attn_type: str,
        metric_name: str,
    ) -> torch.Tensor:
        """
        Calculate the given attention loss and register it as the given metric name.
        """

        assert isinstance(self, TorchGeneratorAgent)
        # Code relies on methods

        # Select the right attention matrices
        selected_student_attn_matrices = [
            layer_matrices[attn_type] for layer_matrices in student_attention_matrices
        ]
        selected_teacher_attn_matrices = [
            layer_matrices[attn_type] for layer_matrices in teacher_attention_matrices
        ]

        batch_size = mask.size(0)
        per_layer_losses = []
        per_layer_per_example_losses = []
        for student_layer_idx, teacher_layer_idx in enumerate(mapped_layers):
            raw_layer_loss = F.mse_loss(
                input=selected_student_attn_matrices[student_layer_idx],
                target=selected_teacher_attn_matrices[teacher_layer_idx],
                reduction='none',
            )
            clamped_layer_loss = torch.clamp(raw_layer_loss, min=0, max=NEAR_INF_FP16)
            # Prevent infs from appearing in the loss term. Especially important with
            # fp16
            reshaped_layer_loss = clamped_layer_loss.view(
                batch_size, -1, clamped_layer_loss.size(-2), clamped_layer_loss.size(-1)
            )
            # [batch size, n heads, query length, key length]
            mean_layer_loss = reshaped_layer_loss.mean(dim=(1, 3))
            # Take the mean over the attention heads and the key length
            assert mean_layer_loss.shape == mask.shape
            masked_layer_loss = mean_layer_loss * mask
            layer_loss_per_example = masked_layer_loss.sum(dim=-1)  # Sum over token dim
            layer_loss = masked_layer_loss.div(num_tokens).sum()
            # Divide before summing over examples so that values don't get too large
            per_layer_losses.append(layer_loss)
            per_layer_per_example_losses.append(layer_loss_per_example)
        attn_loss = torch.stack(per_layer_losses).mean()
        attn_loss_per_example = torch.stack(per_layer_per_example_losses, dim=1).mean(
            dim=1
        )

        # Record metric
        self.record_local_metric(
            metric_name, AverageMetric.many(attn_loss_per_example, tokens_per_example)
        )

        return attn_loss

    def _get_prediction_loss(self, fwd_pass: ForwardPassOutputs) -> torch.Tensor:
        """
        Calculate and return the KL loss on the teacher's prediction layer.

        Also record prediction-loss metrics.
        """
        assert isinstance(self, TorchGeneratorAgent)
        # Code relies on methods
        pred_loss = F.kl_div(
            F.log_softmax(fwd_pass.student_scores, dim=-1, dtype=torch.float),
            F.softmax(fwd_pass.teacher_scores, dim=-1, dtype=torch.float),
            reduction='none',
        ).type_as(fwd_pass.student_scores)
        pred_loss = pred_loss.sum(dim=-1) * fwd_pass.mask
        # Sum over dictionary
        self.record_local_metric(
            'pred_ppl',
            PPLMetric.many(pred_loss.sum(dim=-1), fwd_pass.tokens_per_example),
        )  # Sum over tokens
        self.record_local_metric(
            'pred_loss',
            AverageMetric.many(pred_loss.sum(dim=-1), fwd_pass.tokens_per_example),
        )  # Sum over tokens
        pred_loss = pred_loss.sum() / fwd_pass.num_tokens
        return pred_loss


class DistillTransformerAgentMixin(AbstractDistillTransformerAgentMixin):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group('DistillTransformer arguments')
        agent.add_argument(
            '--copy-teacher-weights',
            type='bool',
            default=True,
            help='Copy weights from the teacher model to the student model',
        )
        return agent

    def __init__(self, opt, shared=None):
        self.copy_teacher_weights = opt['copy_teacher_weights']
        if (
            opt.get('init_model')
            and os.path.isfile(opt['init_model'])
            and self.copy_teacher_weights
        ):
            raise Exception(
                "If --copy-teacher-weights is true, we cannot also copy over weights "
                "with --init-model!"
            )
        super().__init__(opt, shared)
        if shared is None:
            assert self.teacher_agent_opt['n_heads'] == opt['n_heads']

    def build_model(self):

        student_model = super().build_model()

        if self.copy_teacher_weights:

            teacher_model = self._get_teacher_model()

            # Initialize the embeddings
            student_model.encoder.embeddings.load_state_dict(
                teacher_model.encoder.embeddings.state_dict()
            )
            student_model.encoder.position_embeddings.load_state_dict(
                teacher_model.encoder.position_embeddings.state_dict()
            )
            student_model.decoder.embeddings.load_state_dict(
                teacher_model.decoder.embeddings.state_dict()
            )
            student_model.decoder.position_embeddings.load_state_dict(
                teacher_model.decoder.position_embeddings.state_dict()
            )

            # Initialize the encoder and decoder layers
            for student_idx, teacher_idx in enumerate(self.mapped_enc_layers):
                student_model.encoder.layers[student_idx].load_state_dict(
                    teacher_model.encoder.layers[teacher_idx].state_dict()
                )
            for student_idx, teacher_idx in enumerate(self.mapped_dec_layers):
                student_model.decoder.layers[student_idx].load_state_dict(
                    teacher_model.decoder.layers[teacher_idx].state_dict()
                )

        return student_model

    def compute_loss(self, batch, return_output=False):

        fwd_pass = self._perform_forward_passes(batch)

        # Calculate the loss on the encoder's output layer
        encoder_loss = self._get_encoder_loss(fwd_pass)

        # Calculate the loss on the encoder and decoder's hidden states
        enc_hidden_loss, dec_hidden_loss = self._get_hidden_losses(fwd_pass)

        # Calculate the KL loss on the teacher's prediction layer
        pred_loss = self._get_prediction_loss(fwd_pass)

        loss = (
            self.task_loss_coeff * fwd_pass.task_loss
            + self.encoder_loss_coeff * encoder_loss
            + self.hidden_loss_coeff * (enc_hidden_loss + dec_hidden_loss)
            + self.pred_loss_coeff * pred_loss
        )

        if return_output:
            return loss, fwd_pass.student_output
        else:
            return loss


class DistillNarrowTransformerAgentMixin(AbstractDistillTransformerAgentMixin):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group('DistillNarrowTransformer arguments')
        agent.add_argument(
            '--embedding-loss-coeff',
            type=float,
            default=0,
            help='Coefficient on teacher loss on embedding-layer output',
        )
        agent.add_argument(
            '--self-attn-loss-coeff',
            type=float,
            default=0,
            help='Coefficient on teacher loss on self-attention matrices',
        )
        agent.add_argument(
            '--enc-dec-attn-loss-coeff',
            type=float,
            default=0,
            help='Coefficient on teacher loss on enc/dec attention matrices',
        )
        return agent

    def __init__(self, opt, shared=None):
        self.embedding_loss_coeff = opt['embedding_loss_coeff']
        self.self_attn_loss_coeff = opt['self_attn_loss_coeff']
        self.enc_dec_attn_loss_coeff = opt['enc_dec_attn_loss_coeff']
        super().__init__(opt, shared)
        if shared is None:
            assert self.teacher_agent_opt['n_heads'] == opt['n_heads'] or (
                self.self_attn_loss_coeff == 0 and self.enc_dec_attn_loss_coeff == 0
            ), 'The number of attention heads can only differ between the student and teacher models if both attention loss coefficients are 0!'

    def build_model(self):
        student_model = super().build_model()

        # Build projection layers from the student to teacher hidden dim
        student_model.encoder_proj_layer = self._get_projection_layer(student_model)
        student_model.embedding_proj_layers = nn.ModuleDict(
            {
                'encoder': self._get_projection_layer(student_model),
                'decoder': self._get_projection_layer(student_model),
            }
        )
        student_model.hidden_proj_layers = nn.ModuleDict(
            {
                'encoder': nn.ModuleList(
                    [
                        self._get_projection_layer(student_model)
                        for _ in student_model.encoder.layers
                    ]
                ),
                'decoder': nn.ModuleList(
                    [
                        self._get_projection_layer(student_model)
                        for _ in student_model.decoder.layers
                    ]
                ),
            }
        )

        return student_model

    def _get_projection_layer(self, student_model):
        """
        Return a projection layer from the student hidden dim to the teacher hidden dim.
        """

        teacher_model = self._get_teacher_model()

        student_hidden_dim = student_model.encoder.dim
        teacher_hidden_dim = teacher_model.encoder.dim
        assert (
            student_hidden_dim == student_model.decoder.dim
            and teacher_hidden_dim == teacher_model.decoder.dim
        )

        layer = nn.Linear(student_hidden_dim, teacher_hidden_dim)

        # From TinyBERT's BertPreTrainedModel.init_bert_weights() method at
        # https://github.com/huawei-noah/Pretrained-Language-Model/blob/main/TinyBERT/transformer/modeling.py#L628
        layer.weight.data.normal_(mean=0.0, std=0.02)
        layer.bias.data.zero_()

        return layer

    def compute_loss(self, batch, return_output=False):

        fwd_pass = self._perform_forward_passes(batch)

        # Access the student model, which may be wrapped by
        # `torch.nn.parallel.DistributedDataParallel`
        if hasattr(self.model, 'module'):
            student_model = self.model.module
        else:
            student_model = self.model

        # Calculate the loss on the encoder's output layer
        fwd_pass.student_enc_output = student_model.encoder_proj_layer(
            fwd_pass.student_enc_output
        )
        # Pass encoder output through the corresponding projection layer
        encoder_loss = self._get_encoder_loss(fwd_pass)

        # Calculate the loss on the embedding layers
        for module, embedding_output in fwd_pass.student_embedding_outputs.items():
            # Loop over the encoder and the decoder
            fwd_pass.student_embedding_outputs[
                module
            ] = student_model.embedding_proj_layers[module](embedding_output)
            # Pass embedding output through the corresponding projection layer
        enc_emb_loss, dec_emb_loss = self._get_embedding_losses(fwd_pass)

        # Calculate the loss on the encoder and decoder's hidden states
        for module, per_layer_states in fwd_pass.student_hidden_states.items():
            # Loop over the encoder and the decoder
            assert len(per_layer_states) == len(
                student_model.hidden_proj_layers[module]
            )
            for layer_idx, hidden_state in enumerate(per_layer_states):
                # Loop over Transformer layers
                fwd_pass.student_hidden_states[module][
                    layer_idx
                ] = student_model.hidden_proj_layers[module][layer_idx](hidden_state)
                # Pass hidden state through the corresponding projection layer
        enc_hidden_loss, dec_hidden_loss = self._get_hidden_losses(fwd_pass)

        # Calculate the losses on the attention matrices
        if self.self_attn_loss_coeff != 0 or self.enc_dec_attn_loss_coeff != 0:
            (
                enc_self_attn_loss,
                dec_self_attn_loss,
                enc_dec_attn_loss,
            ) = self._get_attention_losses(fwd_pass)
        else:
            # Skip calculating the losses and just set them to 0 because they do not
            # form part of the loss function. This is useful if the number of attention
            # heads is different between the student and teacher, because in that case
            # that the attention query-key matrices will be of different shape.
            enc_self_attn_loss = dec_self_attn_loss = enc_dec_attn_loss = 0

        # Calculate the KL loss on the teacher's prediction layer
        pred_loss = self._get_prediction_loss(fwd_pass)

        loss = (
            self.task_loss_coeff * fwd_pass.task_loss
            + self.encoder_loss_coeff * encoder_loss
            + self.embedding_loss_coeff * (enc_emb_loss + dec_emb_loss)
            + self.hidden_loss_coeff * (enc_hidden_loss + dec_hidden_loss)
            + self.self_attn_loss_coeff * (enc_self_attn_loss + dec_self_attn_loss)
            + self.enc_dec_attn_loss_coeff * (enc_dec_attn_loss)
            + self.pred_loss_coeff * pred_loss
        )

        if return_output:
            return loss, fwd_pass.student_output
        else:
            return loss


class DistillTransformerAgent(DistillTransformerAgentMixin, TransformerGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        DistillTransformerAgentMixin.add_cmdline_args(parser, partial_opt=partial_opt)
        TransformerGeneratorAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser


class DistillNarrowTransformerAgent(
    DistillNarrowTransformerAgentMixin, TransformerGeneratorAgent
):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        DistillNarrowTransformerAgentMixin.add_cmdline_args(
            parser, partial_opt=partial_opt
        )
        TransformerGeneratorAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser


class BartLikeAgent(BartAgent):
    """
    Subclass of the BART agent that prevents loading model weights from bart_large.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        # Just skip BartAgent._initialize_bart(opt)
        super(BartAgent, self).__init__(opt, shared)

    def _manipulate_mask(
        self, mask: torch.BoolTensor, student_scores: torch.Tensor, batch: Batch
    ) -> torch.BoolTensor:
        """
        Add one extra (masked-out) token to the mask, for compatibility with BART.

        Only necessary when examining decoder outputs directly.
        """
        if student_scores.size(1) == batch.label_vec.size(1) + 1:
            mask = torch.cat([mask.new_zeros([mask.size(0), 1]), mask], dim=1)
        return mask


class DistillBartAgent(DistillTransformerAgentMixin, BartLikeAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        DistillTransformerAgentMixin.add_cmdline_args(parser, partial_opt=partial_opt)
        BartLikeAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser


class DistillNarrowBartAgent(DistillNarrowTransformerAgentMixin, BartLikeAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        DistillNarrowTransformerAgentMixin.add_cmdline_args(
            parser, partial_opt=partial_opt
        )
        BartLikeAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser
