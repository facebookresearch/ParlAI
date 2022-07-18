#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
BERT helpers.
"""

from parlai.core.torch_ranker_agent import TorchRankerAgent
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


MODEL_PATH = 'bert-base-uncased.tar.gz'
VOCAB_PATH = 'bert-base-uncased-vocab.txt'


def add_common_args(parser):
    """
    Add command line arguments for this agent.
    """
    TorchRankerAgent.add_cmdline_args(parser, partial_opt=None)
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
    Adds a optional transformer layer and classification layers on top of BERT.
    Args:
        bert_model: pretrained BERT model
        output_dim: dimension of the output layer for defult 1 linear layer classifier. Either output_dim or classifier_layer must be specified
        add_transformer_layer: if additional transformer layer should be added on top of the pretrained model
        layer_pulled: which layer should be pulled from pretrained model
        aggregation: embeddings aggregation (pooling) strategy. Available options are:
            (default)"first" - [CLS] representation,
            "mean" - average of all embeddings except CLS,
            "max" - max of all embeddings except CLS
        classifier_layer: classification layers, can be a signle layer, or list of layers (for ex, torch.nn.Sequential)
    """

    def __init__(
        self,
        bert_model: BertModel,
        output_dim: int = -1,
        add_transformer_layer: bool = False,
        layer_pulled: int = -1,
        aggregation: str = "first",
        classifier_layer: torch.nn.Module = None,
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
        if classifier_layer is None and output_dim == -1:
            raise Exception(
                "Either output dimention or classifier layers must be specified"
            )
        elif classifier_layer is None:
            self.additional_linear_layer = torch.nn.Linear(bert_output_dim, output_dim)
        else:
            self.additional_linear_layer = classifier_layer
            if output_dim != -1:
                print(
                    "Both classifier layer and output dimension are specified. Output dimension parameter is ignored."
                )
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
        result = result + 0 * torch.sum(output_pooler)
        return result


def surround(idx_vector, start_idx, end_idx):
    """
    Surround the vector by start_idx and end_idx.
    """
    start_tensor = idx_vector.new_tensor([start_idx])
    end_tensor = idx_vector.new_tensor([end_idx])
    return torch.cat([start_tensor, idx_vector, end_tensor], 0)
