#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Modules for TransresnetMultimodalAgent.
"""

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import torch
from torch import nn
from parlai.utils.io import PathManager
from parlai.agents.transformer.modules import (
    TransformerEncoder,
    create_position_codes,
    TransformerEncoderLayer,
)
from projects.personality_captions.transresnet.modules import (
    TransresnetModel,
    load_fasttext_embeddings,
)


class TransresnetMultimodalModel(TransresnetModel):
    """
    Extension of Transresnet to incorporate dialogue history and multimodality.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Override to include model-specific args.
        """
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group("TransresnetMultimodal task arguments")
        agent.add_argument(
            "--context-encoder-embedding-type",
            type=str,
            default=None,
            choices=[None, "fasttext_cc"],
            help="Specify if using pretrained embeddings",
        )
        agent.add_argument(
            "--load-context-encoder-from",
            type=str,
            default=None,
            help="Specify if using a pretrained transformer encoder",
        )
        agent.add_argument(
            "--share-encoder",
            type="bool",
            default=False,
            help="Whether to share the text encoder for the "
            "labels and the dialogue history",
        )
        agent.add_argument("--num-layers-multimodal-encoder", type=int, default=1)
        agent.add_argument(
            "--multimodal",
            type="bool",
            default=False,
            help="If true, feed a query term into a separate "
            "transformer prior to computing final rank "
            "scores",
        )
        agent.add_argument(
            "--multimodal-combo",
            type=str,
            choices=["concat", "sum"],
            default="sum",
            help="How to combine the encoding for the " "multi-modal transformer",
        )
        agent.add_argument(
            "--encode-image",
            type="bool",
            default=True,
            help="Whether to include the image encoding when "
            "retrieving a candidate response",
        )
        agent.add_argument(
            "--encode-dialogue-history",
            type="bool",
            default=True,
            help="Whether to include the dialogue history "
            "encoding when retrieving a candidate response",
        )
        agent.add_argument(
            "--encode-personality",
            type="bool",
            default=True,
            help="Whether to include the personality encoding "
            "when retrieving a candidate response",
        )
        return parser

    def __init__(self, opt, personalities_list, dictionary):
        super().__init__(opt, personalities_list, dictionary)
        self.hidden_dim = self.opt["hidden_dim"]
        self.share_encoder = opt.get("share_encoder")
        nlayers_mm = (
            opt["num_layers_all"]
            if opt["num_layers_all"] != -1
            else opt["num_layers_multimodal_encoder"]
        )

        # blank encoding (for concat)
        self.blank_encoding = torch.Tensor(opt["hidden_dim"]).fill_(0).detach_()
        if self.use_cuda:
            self.blank_encoding = self.blank_encoding.cuda()

        # Encoders
        self.encode_image = opt.get("encode_image", True)
        self.encode_personality = opt.get("encode_personality", True)
        self.encode_dialogue_history = opt.get("encode_dialogue_history", True)
        assert any(
            [self.encode_dialogue_history, self.encode_image, self.encode_personality]
        )

        # Transformer 2
        self._build_multimodal_encoder(nlayers_mm)

        # Label Encoder
        self.label_encoder = self.text_encoder

        # Context encoder
        self._build_context_encoder()

    def _build_multimodal_encoder(self, n_layers_mm):
        """
        Build the multimodal encoder.

        :param n_layers_mm:
            number of layers for the transformer
        """
        self.multimodal = self.opt.get("multimodal")
        if self.multimodal:
            self.multimodal_combo = self.opt.get("multimodal_combo", "sum")
            nlayers_mm = (
                self.opt["num_layers_all"]
                if self.opt["num_layers_all"] != -1
                else self.opt["num_layers_multimodal_encoder"]
            )
            self.multimodal_encoder = MultimodalCombiner(
                n_heads=self.opt["n_heads"],
                n_layers=nlayers_mm,
                hidden_dim=self.opt["hidden_dim"],
                ffn_size=self.opt["embedding_size"] * 4,
                attention_dropout=self.opt["attention_dropout"],
                relu_dropout=self.opt["relu_dropout"],
                learn_positional_embeddings=self.opt.get(
                    "learn_positional_embeddings", False
                ),
                reduction=True,
            )

    def _build_context_encoder(self):
        """
        Build the context (i.e. dialogue history) encoder.
        """
        if self.opt.get("share_encoder"):
            self.context_encoder = self.label_encoder
        else:
            if (
                self.opt["load_context_encoder_from"] is None
                and self.opt["context_encoder_embedding_type"] == "fasttext_cc"
            ):
                embeddings = load_fasttext_embeddings(
                    self.dictionary, self.opt["embedding_size"], self.opt["datapath"]
                )
            else:
                embeddings = nn.Embedding(
                    len(self.dictionary), self.opt["embedding_size"]
                )
            self.context_encoder = TransformerEncoder(
                opt=self.opt,
                embedding=embeddings,
                vocabulary_size=len(self.dictionary),
                padding_idx=self.dictionary.tok2ind[self.dictionary.null_token],
                embeddings_scale=False,
                output_scaling=1.0,
            )
            if self.opt.get("load_context_encoder_from") is not None:
                self._load_context_encoder_state()

    def forward(
        self,
        image_features,
        personalities,
        dialogue_histories,
        labels,
        batchsize=None,
        personalities_tensor=None,
    ):
        """
        Model forward pass.

        :param image_features:
            list of tensors of image features, one per example
        :param personalities:
            list of personalities, one per example
        :param dialogue_histories:
            list of dialogue histories, one per example
        :param labels:
            list of response labels, one per example
        :param personalities_tensor:
            (optional) list of personality representations, usually a one-hot
            vector if specified

        :return:
            the encoded context and the encoded captions.
        """
        # labels
        labels_encoded = self.forward_text_encoder(labels)
        # dialog history
        d_hist_encoded = self.forward_text_encoder(
            dialogue_histories, dialogue_history=True, batchsize=batchsize
        )
        # images
        img_encoded = self.forward_image(image_features)
        # personalities
        pers_encoded = self.forward_personality(personalities, personalities_tensor)
        total_encoded = self.get_rep(
            [img_encoded, d_hist_encoded, pers_encoded], batchsize=batchsize
        )
        loss, nb_ok = self.get_loss(total_encoded, labels_encoded)

        return loss, nb_ok, total_encoded

    def forward_personality(self, personalities, personalities_tensor):
        """
        Encode personalities.

        :param personalities:
            list of personalities, one per example
        :param personalities_tensor:
            (optional) list of personality representations, usually a one-hot
            vector if specified

        :return:
            encoded representation of the personalities
        """
        pers_encoded = None
        if not self.encode_personality:
            if self.multimodal and self.multimodal_combo == "concat":
                pers_encoded = self.blank_encoding
        else:
            pers_encoded = super().forward_personality(
                personalities, personalities_tensor
            )
        return pers_encoded

    def forward_text_encoder(self, texts, dialogue_history=False, batchsize=None):
        """
        Forward pass for a text encoder.

        :param texts:
            text to encode
        :param dialogue_history:
            flag that indicates whether the text is dialogue history; if False,
            text is a response candidate
        :param batchsize:
            size of the batch

        :return:
            encoded representation of the `texts`
        """
        texts_encoded = None
        if texts is None or (dialogue_history and not self.encode_dialogue_history):
            if (
                self.multimodal
                and self.multimodal_combo == "concat"
                and dialogue_history
            ):
                texts_encoded = torch.stack(
                    [self.blank_encoding for _ in range(batchsize)]
                )
        else:
            encoder = self.context_encoder if dialogue_history else self.label_encoder
            indexes, mask = self.captions_to_tensor(texts)
            texts_encoded = encoder(indexes)
            if self.text_encoder_frozen:
                texts_encoded = texts_encoded.detach()
            texts_encoded = self.additional_layer(texts_encoded)

        return texts_encoded

    def forward_image(self, image_features):
        """
        Encode image features.

        :param image_features:
            list of image features

        :return:
            encoded representation of the image features
        """
        img_encoded = None
        if image_features is None or not self.encode_image:
            if self.multimodal and self.multimodal_combo == "concat":
                img_encoded = self.blank_encoding
        else:
            img_encoded = super().forward_image(image_features)

        return img_encoded

    def get_rep(self, encodings, batchsize=None):
        """
        Get the multimodal representation of the encodings.

        :param encodings:
            list of encodings
        :param batchsize:
            size of batch

        :return:
            final multimodal representations
        """
        if not self.multimodal:
            rep = self.sum_encodings(encodings)
        else:
            if self.multimodal_combo == "sum":
                encodings = self.sum_encodings(encodings).unsqueeze(1)
            elif self.multimodal_combo == "concat":
                encodings = self.cat_encodings(encodings)
            all_one_mask = torch.ones(encodings.size()[:2])
            if self.use_cuda:
                all_one_mask = all_one_mask.cuda()
            rep = self.multimodal_encoder(encodings, all_one_mask)
        if rep is None:
            rep = torch.stack([self.blank_encoding for _ in range(batchsize)])
        return rep

    def choose_best_response(
        self,
        image_features,
        personalities,
        dialogue_histories,
        candidates,
        candidates_encoded=None,
        k=1,
        batchsize=None,
    ):
        """
        Choose the best response for each example.

        :param image_features:
            list of tensors of image features
        :param personalities:
            list of personalities
        :param dialogue_histories:
            list of dialogue histories, one per example
        :param candidates:
            list of candidates, one set per example
        :param candidates_encoded:
            optional; if specified, a fixed set of encoded candidates that is
            used for each example
        :param k:
            number of ranked candidates to return. if < 1, we return the ranks
            of all candidates in the set.

        :return:
            a set of ranked candidates for each example
        """
        self.eval()
        _, _, encoded = self.forward(
            image_features, personalities, dialogue_histories, None, batchsize=batchsize
        )
        encoded = encoded.detach()
        one_cand_set = True
        if candidates_encoded is None:
            one_cand_set = False
            candidates_encoded = [
                self.forward_text_encoder(c).detach() for c in candidates
            ]
        chosen = [
            self.choose_topk(
                idx if not one_cand_set else 0,
                encoded,
                candidates,
                candidates_encoded,
                one_cand_set,
                k,
            )
            for idx in range(len(encoded))
        ]
        return chosen

    def choose_topk(
        self, idx, encoded, candidates, candidates_encoded, one_cand_set, k
    ):
        """
        Choose top k best responses for a single example.

        :param idx:
            idx of example in encoded
        :param encoded:
            full matrix of encoded representations (for the whole batch)
        :param candidates:
            list of candidates
        :param candidates_encoded:
            encoding of the candidates
        :param one_cand_set:
            true if there is one set of candidates for each example
        :param k:
            how many ranked responses to return

        :return:
            ranked list of k responses
        """
        encoding = encoded[idx : idx + 1, :]
        scores = torch.mm(
            candidates_encoded[idx] if not one_cand_set else candidates_encoded,
            encoding.transpose(0, 1),
        )
        if k >= 1:
            _, index_top = torch.topk(scores, k, dim=0)
        else:
            _, index_top = torch.topk(scores, scores.size(0), dim=0)
        return [
            candidates[idx][idx2] if not one_cand_set else candidates[idx2]
            for idx2 in index_top.unsqueeze(1)
        ]

    def get_loss(self, total_encoded, labels_encoded):
        """
        Compute loss over batch.

        :param total_encoded:
            encoding of the examples
        :param labels_encoded:
            encoding of the labels

        :return:
            total batch loss, and number of correct examples
        """
        loss = None
        num_correct = None
        if labels_encoded is not None:
            dot_products = total_encoded.mm(
                labels_encoded.t()
            )  # batch_size * batch_size
            log_prob = torch.nn.functional.log_softmax(dot_products, dim=1)
            targets = torch.arange(0, len(total_encoded), dtype=torch.long)
            if self.use_cuda:
                targets = targets.cuda()
            loss = torch.nn.functional.nll_loss(log_prob, targets)
            num_correct = (log_prob.max(dim=1)[1] == targets).float().sum()
        return loss, num_correct

    def cat_encodings(self, tensors):
        """
        Concatenate non-`None` encodings.

        :param tensors:
            list tensors to concatenate

        :return:
            concatenated tensors
        """
        tensors = [t for t in tensors if t is not None]
        return torch.cat([t.unsqueeze(1) for t in tensors], dim=1)

    def _load_text_encoder_state(self):
        try:
            state_file = self.opt.get("load_encoder_from")
            with PathManager.open(state_file, 'rb') as f:
                model = torch.load(f)
            states = model["model"]
            self.text_encoder.load_state_dict(states)
        except Exception as e:
            print(
                "WARNING: Cannot load transformer state; please make sure "
                "specified file is a dictionary with the states in `model`. "
                "Additionally, make sure that the appropriate options are "
                "specified. Error: {}".format(e)
            )

    def _load_context_encoder_state(self):
        try:
            state_file = self.opt.get("load_context_encoder_from")
            with PathManager.open(state_file, 'rb') as f:
                model = torch.load(f)
            states = model["model"]
            self.context_encoder.load_state_dict(states)
        except Exception as e:
            print(
                "WARNING: Cannot load transformer state; please make sure "
                "specified file is a dictionary with the states in `model`. "
                "Additionally, make sure that the appropriate options are "
                "specified. Error: {}".format(e)
            )


class MultimodalCombiner(nn.Module):
    """
    Multimodal Combination module.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        hidden_dim,
        ffn_size,
        reduction=True,
        attention_dropout=0.0,
        relu_dropout=0.0,
        learn_positional_embeddings=False,
    ):
        super().__init__()
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.out_dim = hidden_dim
        self.dim = hidden_dim
        self.reduction = reduction
        assert hidden_dim % n_heads == 0, "MM-Combiner dim must be multiple of n_heads"
        n_positions = 1024
        self.position_embeddings = nn.Embedding(n_positions, hidden_dim)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, hidden_dim, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, hidden_dim**-0.5)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    n_heads, hidden_dim, ffn_size, attention_dropout, relu_dropout
                )
            )

    def forward(self, tensor, mask):
        """
        Forward pass.

        :param tensor:
            a [bsz, seq_len, hidden_dim] FloatTensor
        :param mask:
            a [bsz, seq_len] ByteTensor filled with 1 when inside the sequence and 0 outside.

        :return:
            output: a [bsz, hidden_dim] FloatTensor of encodings
            mask: the same as before
        """
        seq_len = tensor.size(1)
        positions = tensor.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        tensor *= mask.unsqueeze(-1).float()
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)

        if self.reduction:
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1e-20)
            output = tensor.sum(dim=1) / divisor
            return output
        else:
            output = tensor
            return output, mask
