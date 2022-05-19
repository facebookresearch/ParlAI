#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
BERT classifier agent uses bert embeddings to make an utterance-level classification.

This implementation allows to customize classifier layers with input arguments.
"""

import os
from collections import deque
from typing import Optional

import torch
from parlai.agents.bert_ranker.bert_dictionary import BertDictionaryAgent
from parlai.agents.bert_ranker.helpers import BertWrapper, MODEL_PATH
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import History
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.utils.io import PathManager
from parlai.utils.misc import warn_once
from parlai.zoo.bert.build import download

try:
    from pytorch_pretrained_bert import BertModel
except ImportError:
    raise ImportError(
        "BERT rankers needs pytorch-pretrained-BERT installed. \n "
        "pip install pytorch-pretrained-bert"
    )


LINEAR = "linear"
RELU = "relu"

SUPPORTED_LAYERS = [LINEAR, RELU]


class BertClassifierHistory(History):
    """
    Handles tokenization history.
    """

    def __init__(self, opt, **kwargs):
        self.sep_last_utt = opt.get("sep_last_utt", False)
        super().__init__(opt, **kwargs)

    def get_history_vec(self):
        """
        Override from parent class to possibly add [SEP] token.
        """
        if not self.sep_last_utt or len(self.history_vecs) <= 1:
            return super().get_history_vec()

        history = deque(maxlen=self.max_len)
        for vec in self.history_vecs[:-1]:
            history.extend(vec)
            history.extend(self.delimiter_tok)
        history.extend([self.dict.end_idx])  # add [SEP] token
        history.extend(self.history_vecs[-1])

        return history


class BertClassifierAgent(TorchClassifierAgent):
    """
    Classifier based on BERT implementation.
    """

    def __init__(self, opt, shared=None):
        # download pretrained models
        download(opt["datapath"])
        self.pretrained_path = PathManager.get_local_path(
            os.path.join(opt["datapath"], "models", "bert_models", MODEL_PATH)
        )
        opt["pretrained_path"] = self.pretrained_path
        self.add_cls_token = opt.get("add_cls_token", True)
        self.sep_last_utt = opt.get("sep_last_utt", False)
        self.classifier_layers = opt.get("classifier_layers", None)
        super().__init__(opt, shared)

    @classmethod
    def history_class(cls):
        """
        Determine the history class.
        """
        return BertClassifierHistory

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add CLI args.
        """
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        parser = parser.add_argument_group("BERT Classifier Arguments")
        parser.add_argument(
            "--add-cls-token",
            type="bool",
            default=True,
            help="add [CLS] token to text vec",
        )
        parser.add_argument(
            "--sep-last-utt",
            type="bool",
            default=False,
            help="separate the last utterance into a different"
            "segment with [SEP] token in between",
        )
        parser.add_argument(
            "--classifier-layers",
            nargs='+',
            type=str,
            default=None,
            help="list of classifier layers comma-separated with layer's dimension where applicable. For example: linear,64 linear,32 relu",
        )
        parser.set_defaults(dict_maxexs=0)  # skip building dictionary
        return parser

    @staticmethod
    def dictionary_class():
        """
        Determine the dictionary class.
        """
        return BertDictionaryAgent

    @classmethod
    def upgrade_opt(cls, opt_on_disk):
        """
        Upgrade opts from older model files.
        """
        super(BertClassifierAgent, cls).upgrade_opt(opt_on_disk)

        # 2019-06-25: previous versions of the model did not add a CLS token
        # to the beginning of text_vec.
        if "add_cls_token" not in opt_on_disk:
            warn_once("Old model: overriding `add_cls_token` to False.")
            opt_on_disk["add_cls_token"] = False

        return opt_on_disk

    def _get_layer_parameters(self, prev_dimension, output_dimension):
        """
        Parse layer definitions from the input.
        """
        layers = []
        dimensions = []
        no_dimension = -1
        for layer in self.classifier_layers:
            if ',' in layer:
                l, d = layer.split(',')
                layers.append(l)
                dimensions.append((prev_dimension, int(d)))
                prev_dimension = int(d)
            else:
                layers.append(layer)
                dimensions.append(no_dimension)
        ind = 0
        while (
            ind < len(dimensions)
            and dimensions[len(dimensions) - ind - 1] == no_dimension
        ):
            ind += 1
        if (ind == len(dimensions) and prev_dimension == output_dimension) or (
            ind < len(dimensions) and dimensions[ind][1] == output_dimension
        ):
            return layers, dimensions

        if ind < len(dimensions):
            raise Exception(
                f"Output layer's dimension does not match number of classes. Found {dimensions[ind][1]}, expected {output_dimension}"
            )
        raise Exception(
            f"Output layer's dimension does not match number of classes. Found {prev_dimension}, expected {output_dimension}"
        )

    def _map_layer(self, layer: str, dim=None):
        """
        Get torch wrappers for nn layers.
        """
        if layer == LINEAR:
            return torch.nn.Linear(dim[0], dim[1])
        elif layer == RELU:
            return torch.nn.ReLU(inplace=False)
        raise Exception(
            "Unrecognized network layer {}. Available options are: {}".format(
                layer, ", ".join(SUPPORTED_LAYERS)
            )
        )

    def build_model(self):
        """
        Construct the model.
        """
        num_classes = len(self.class_list)
        bert_model = BertModel.from_pretrained(self.pretrained_path)
        if self.classifier_layers is not None:
            prev_dimension = bert_model.embeddings.word_embeddings.weight.size(1)
            layers, dims = self._get_layer_parameters(
                prev_dimension=prev_dimension, output_dimension=num_classes
            )
            decoders = torch.nn.Sequential()
            for l, d in zip(layers, dims):
                decoders.append(self._map_layer(l, d))
            return BertWrapper(bert_model=bert_model, classifier_layer=decoders)
        return BertWrapper(bert_model=bert_model, output_dim=num_classes)

    def _set_text_vec(self, *args, **kwargs):
        obs = super()._set_text_vec(*args, **kwargs)
        if "text_vec" in obs and self.add_cls_token:
            # insert [CLS] token
            if "added_start_end_tokens" not in obs:
                # Sometimes the obs is cached (meaning its the same object
                # passed the next time) and if so, we would continually re-add
                # the start/end tokens. So, we need to test if already done
                start_tensor = torch.LongTensor([self.dict.start_idx])
                new_text_vec = torch.cat([start_tensor, obs["text_vec"]], 0)
                obs.force_set("text_vec", new_text_vec)
                obs["added_start_end_tokens"] = True
        return obs

    def score(self, batch):
        """
        Score the batch.
        """
        segment_idx = (batch.text_vec * 0).long()
        if self.sep_last_utt:
            batch_len = batch.text_vec.size(1)
            # find where [SEP] token is
            seps = (batch.text_vec == self.dict.end_idx).nonzero()
            if len(seps) > 0:
                for row in seps:
                    # set last utterance to segment 1
                    segment_idx[row[0], list(range(row[1], batch_len))] = 1
            else:
                # only one utterance: everything after [CLS] token
                # should be segment 1
                segment_idx = (batch.text_vec != self.dict.start_idx).long()
        mask = (batch.text_vec != self.NULL_IDX).long()
        token_idx = batch.text_vec * mask
        return self.model(token_idx, segment_idx, mask)
