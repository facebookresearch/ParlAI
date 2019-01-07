#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.torch_ranker_agent import TorchRankerAgent
import torch
try:
    from pytorch_pretrained_bert.modeling import BertLayer, BertConfig
    from pytorch_pretrained_bert import BertAdam
except ImportError:
    raise Exception(("BERT rankers needs pytorch-pretrained-BERT installed. \n "
                     "pip install pytorch-pretrained-bert"))
from parlai.core.utils import _ellipse


def add_common_args(parser):
    """Add command line arguments for this agent."""
    TorchRankerAgent.add_cmdline_args(parser)
    parser = parser.add_argument_group('Bert Ranker Arguments')
    parser.add_argument('--history-length', type=int, default=5,
                        help='Number of previous line to keep for inference')
    parser.add_argument('--num-samples', type=int, default=131800,
                        help='Number of samples in the task (temporary)')
    parser.add_argument('--bert-id', type=str, default='bert-base-uncased')
    parser.add_argument(
        '--add-transformer-layer',
        type="bool",
        default=False,
        help="Also add a transformer layer on top of Bert")
    parser.add_argument('--pull-from-layer', type=int, default=-1,
                        help="Which layer of Bert do we use? Default=-1=last one.")
    parser.add_argument('--predefined-candidates-path', type=str, default=None,
                        help="Path to a list of candidates")
    parser.add_argument('--token-cap', type=int, default=320,
                        help="Cap number of tokens")
    parser.add_argument('--out-dim', type=int, default=768,
                        help="For biencoder, output dimension")
    parser.add_argument('--topn', type=int, default=10,
                        help="For the biencoder: select how many elements to return")
    parser.add_argument(
        '--type-optimization',
        type=str,
        default="additional_layers",
        choices=[
            "additional_layers",
            "top_layer",
            "top4_layers",
            "all_encoder_layers",
            "all"],
        help="Which part of the encoders do we optimize. (Default: the top one.)")


class BertWrapper(torch.nn.Module):
    """
        Adds a optional transformer layer and a linear layer on top of Bert
    """

    def __init__(self, bert_model, output_dim,
                 add_transformer_layer=False, layer_pulled=-1):
        super(BertWrapper, self).__init__()
        self.layer_pulled = layer_pulled
        self.add_transformer_layer = add_transformer_layer
        # deduce bert output dim from the size of the pooler in bert
        bert_output_dim = bert_model.pooler.dense.weight.size()[0]

        if add_transformer_layer:
            config_for_one_layer = BertConfig(
                0, hidden_size=bert_output_dim, num_attention_heads=int(
                    bert_output_dim / 64), intermediate_size=3072, hidden_act="gelu")
            self.additional_transformer_layer = BertLayer(config_for_one_layer)
        self.additional_linear_layer = torch.nn.Linear(bert_output_dim, output_dim)
        self.bert_model = bert_model

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert, _ = self.bert_model(token_ids, segment_ids, attention_mask)
        # output_bert is a list of 12 (for bert base) layers.
        layer_of_interest = output_bert[self.layer_pulled]
        if self.add_transformer_layer:
            # Follow up by yet another transformer layer
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            embeddings = self.additional_transformer_layer(
                layer_of_interest, extended_attention_mask)[:, 0, :]
        else:
            embeddings = layer_of_interest[:, 0, :]
        return self.additional_linear_layer(embeddings)


def surround(idx_vector, start_idx, end_idx):
    """ Surround the vector by start_idx and end_idx
    """
    start_tensor = idx_vector.new_tensor([start_idx])
    end_tensor = idx_vector.new_tensor([end_idx])
    return torch.cat([start_tensor, idx_vector, end_tensor], 0)


patterns_optimizer = {
    "additional_layers": ["additional"],
    "top_layer": [
        "additional",
        "bert_model.encoder.layer.11."],
    "top4_layers": [
        "additional",
        "bert_model.encoder.layer.11.",
        "encoder.layer.10.",
        "encoder.layer.9.",
        "encoder.layer.8"],
    "all_encoder_layers": [
        "additional",
        "bert_model.encoder.layer"],
    "all": [
        "additional",
        "bert_model.encoder.layer",
        "bert_model.embeddings"],
}


def get_bert_optimizer(models, type_optimization, number_iterations, proportion_warmup,
                       learning_rate):
    """
        Provides an optimizer already configured with weight decay for the parameters
        that need it. Weight decay is left standard at 0.01
    """
    if type_optimization not in patterns_optimizer:
        print("Error. Type optimizer must be one of %s" %
              (str(patterns_optimizer.keys())))
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

    print("The following parameters will be optimized WITH decay:")
    print(_ellipse(parameters_with_decay_names, 5, " , "))
    print("The following parameters will be optimized WITHOUT decay:")
    print(_ellipse(parameters_without_decay_names, 5, " , "))

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay_rate': 0.01},
        {'params': parameters_without_decay, 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=proportion_warmup,
                         t_total=number_iterations)
    return optimizer
