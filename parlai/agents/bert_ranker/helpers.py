#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
BERT helpers.
"""

from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.fp16 import fp16_optimizer_wrapper
from parlai.utils.torch import neginf

try:
    from pytorch_pretrained_bert.modeling import BertLayer, BertConfig
    from pytorch_pretrained_bert import BertModel  # NOQA
except ImportError:
    raise ImportError(
        'This model requires that huggingface\'s transformers is '
        'installed. Install with:\n `pip install transformers`.'
    )


import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_


MODEL_PATH = 'bert-base-uncased.tar.gz'
VOCAB_PATH = 'bert-base-uncased-vocab.txt'


def add_common_args(parser):
    """
    Add command line arguments for this agent.
    """
    TorchRankerAgent.add_cmdline_args(parser)
    parser = parser.add_argument_group('Bert Ranker Arguments')
    parser.add_argument(
        '--add-transformer-layer',
        type='bool',
        default=False,
        help='Also add a transformer layer on top of Bert',
    )
    parser.add_argument(
        '--pull-from-layer',
        type=int,
        default=-1,
        help='Which layer of Bert do we use? Default=-1=last one.',
    )
    parser.add_argument(
        '--out-dim', type=int, default=768, help='For biencoder, output dimension'
    )
    parser.add_argument(
        '--topn',
        type=int,
        default=10,
        help='For the biencoder: select how many elements to return',
    )
    parser.add_argument(
        '--data-parallel',
        type='bool',
        default=False,
        help='use model in data parallel, requires '
        'multiple gpus. NOTE This is incompatible'
        ' with distributed training',
    )
    parser.add_argument(
        '--type-optimization',
        type=str,
        default='all_encoder_layers',
        choices=[
            'additional_layers',
            'top_layer',
            'top4_layers',
            'all_encoder_layers',
            'all',
        ],
        help='Which part of the encoders do we optimize. '
        '(Default: all_encoder_layers.)',
    )
    parser.add_argument(
        '--bert-aggregation',
        type=str,
        default='first',
        choices=['first', 'max', 'mean'],
        help='How do we transform a list of output into one',
    )
    parser.set_defaults(
        label_truncate=300,
        text_truncate=300,
        learningrate=0.00005,
        eval_candidates='inline',
        candidates='batch',
        dict_maxexs=0,  # skip building dictionary
    )


class BertWrapper(torch.nn.Module):
    """
    Adds a optional transformer layer and a linear layer on top of BERT.
    """

    def __init__(
        self,
        bert_model,
        output_dim,
        add_transformer_layer=False,
        layer_pulled=-1,
        aggregation="first",
    ):
        super(BertWrapper, self).__init__()
        self.layer_pulled = layer_pulled
        self.aggregation = aggregation
        self.add_transformer_layer = add_transformer_layer
        # deduce bert output dim from the size of embeddings
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        if add_transformer_layer:
            config_for_one_layer = BertConfig(
                0,
                hidden_size=bert_output_dim,
                num_attention_heads=int(bert_output_dim / 64),
                intermediate_size=3072,
                hidden_act='gelu',
            )
            self.additional_transformer_layer = BertLayer(config_for_one_layer)
        self.additional_linear_layer = torch.nn.Linear(bert_output_dim, output_dim)
        self.bert_model = bert_model

    def forward(self, token_ids, segment_ids, attention_mask):
        """
        Forward pass.
        """
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # output_bert is a list of 12 (for bert base) layers.
        layer_of_interest = output_bert[self.layer_pulled]
        dtype = next(self.parameters()).dtype
        if self.add_transformer_layer:
            # Follow up by yet another transformer layer
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (~extended_attention_mask).to(dtype) * neginf(
                dtype
            )
            embedding_layer = self.additional_transformer_layer(
                layer_of_interest, extended_attention_mask
            )
        else:
            embedding_layer = layer_of_interest

        if self.aggregation == "mean":
            #  consider the average of all the output except CLS.
            # obviously ignores masked elements
            outputs_of_interest = embedding_layer[:, 1:, :]
            mask = attention_mask[:, 1:].type_as(embedding_layer).unsqueeze(2)
            sumed_embeddings = torch.sum(outputs_of_interest * mask, dim=1)
            nb_elems = torch.sum(attention_mask[:, 1:].type(dtype), dim=1).unsqueeze(1)
            embeddings = sumed_embeddings / nb_elems
        elif self.aggregation == "max":
            #  consider the max of all the output except CLS
            outputs_of_interest = embedding_layer[:, 1:, :]
            mask = (~attention_mask[:, 1:]).type(dtype).unsqueeze(2) * neginf(dtype)
            embeddings, _ = torch.max(outputs_of_interest + mask, dim=1)
        else:
            # easiest, we consider the output of "CLS" as the embedding
            embeddings = embedding_layer[:, 0, :]

        # We need this in case of dimensionality reduction
        result = self.additional_linear_layer(embeddings)

        # Sort of hack to make it work with distributed: this way the pooler layer
        # is used for grad computation, even though it does not change anything...
        # in practice, it just adds a very (768*768) x (768*batchsize) matmul
        result += 0 * torch.sum(output_pooler)
        return result


def surround(idx_vector, start_idx, end_idx):
    """
    Surround the vector by start_idx and end_idx.
    """
    start_tensor = idx_vector.new_tensor([start_idx])
    end_tensor = idx_vector.new_tensor([end_idx])
    return torch.cat([start_tensor, idx_vector, end_tensor], 0)


patterns_optimizer = {
    'additional_layers': ['additional'],
    'top_layer': ['additional', 'bert_model.encoder.layer.11.'],
    'top4_layers': [
        'additional',
        'bert_model.encoder.layer.11.',
        'encoder.layer.10.',
        'encoder.layer.9.',
        'encoder.layer.8',
    ],
    'all_encoder_layers': ['additional', 'bert_model.encoder.layer'],
    'all': ['additional', 'bert_model.encoder.layer', 'bert_model.embeddings'],
}


def get_bert_optimizer(models, type_optimization, learning_rate, fp16=False):
    """
    Optimizes the network with AdamWithDecay.
    """
    if type_optimization not in patterns_optimizer:
        print(
            'Error. Type optimizer must be one of %s' % (str(patterns_optimizer.keys()))
        )
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    no_decay = ['bias', 'gamma', 'beta']
    patterns = patterns_optimizer[type_optimization]

    for model in models:
        for n, p in model.named_parameters():
            if any(t in n for t in patterns):
                if any(t in n for t in no_decay):
                    parameters_without_decay.append(p)
                    parameters_without_decay_names.append(n)
                else:
                    parameters_with_decay.append(p)
                    parameters_with_decay_names.append(n)

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay': 0.01},
        {'params': parameters_without_decay, 'weight_decay': 0.0},
    ]
    optimizer = AdamWithDecay(optimizer_grouped_parameters, lr=learning_rate)

    if fp16:
        optimizer = fp16_optimizer_wrapper(optimizer)

    return optimizer


# TODO: deprecate this entire class; it should be subsumed by TA as of pytorch 1.2
class AdamWithDecay(Optimizer):
    """
    Adam with decay; mirror's HF's implementation.

    :param lr:
        learning rate
    :param b1:
        Adams b1. Default: 0.9
    :param b2:
        Adams b2. Default: 0.999
    :param e:
        Adams epsilon. Default: 1e-6
    :param weight_decay:
        Weight decay. Default: 0.01
    :param max_grad_norm:
        Maximum norm for the gradients (-1 means no clipping).  Default: 1.0
    """

    def __init__(
        self, params, lr, b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0
    ):
        if lr < 0.0:
            raise ValueError('Invalid learning rate: {} - should be >= 0.0'.format(lr))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                'Invalid b1 parameter: {} - should be in [0.0, 1.0['.format(b1)
            )
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                'Invalid b2 parameter: {} - should be in [0.0, 1.0['.format(b2)
            )
        if not e >= 0.0:
            raise ValueError('Invalid epsilon value: {} - should be >= 0.0'.format(e))
        defaults = dict(
            lr=lr,
            b1=b1,
            b2=b2,
            e=e,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super(AdamWithDecay, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        :param closure:
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please '
                        'consider SparseAdam instead'
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data
                lr = group['lr']

                update_with_lr = lr * update
                p.data.add_(-update_with_lr)
        return loss
