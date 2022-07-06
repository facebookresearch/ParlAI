#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import torch
from parlai.agents.seq2seq.modules import opt_to_kwargs
from parlai.core.torch_generator_agent import TorchGeneratorAgent

from .modules import HredModel


class HredAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group("HRED Arguments")
        agent.add_argument(
            "-hs",
            "--hiddensize",
            type=int,
            default=128,
            help="size of the hidden layers",
        )
        agent.add_argument(
            "-esz",
            "--embeddingsize",
            type=int,
            default=128,
            help="size of the token embeddings",
        )
        agent.add_argument(
            "-nl", "--numlayers", type=int, default=2, help="number of hidden layers"
        )
        agent.add_argument(
            "-dr", "--dropout", type=float, default=0.1, help="dropout rate"
        )
        agent.add_argument(
            "-lt",
            "--lookuptable",
            default="unique",
            choices=["unique", "enc_dec", "dec_out", "all"],
            help="The encoder, decoder, and output modules can "
            "share weights, or not. "
            "Unique has independent embeddings for each. "
            "Enc_dec shares the embedding for the encoder "
            "and decoder. "
            "Dec_out shares decoder embedding and output "
            "weights. "
            "All shares all three weights.",
        )
        agent.add_argument(
            "-idr",
            "--input-dropout",
            type=float,
            default=0.0,
            help="Probability of replacing tokens with UNK in training.",
        )

        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return agent

    def __init__(self, opt, shared=None):
        """
        Set up model.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        super().__init__(opt, shared)
        self.id = "Hred"

    def build_model(self, states=None):
        opt = self.opt
        if not states:
            states = {}
        kwargs = opt_to_kwargs(opt)

        model = HredModel(
            len(self.dict),
            opt["embeddingsize"],
            opt["hiddensize"],
            device=self.device,
            padding_idx=self.NULL_IDX,
            start_idx=self.START_IDX,
            end_idx=self.END_IDX,
            unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get("longest_label", 1),
            **kwargs,
        )

        if opt.get("dict_tokenizer") == "bpe" and opt["embedding_type"] != "random":
            print("skipping preinitialization of embeddings for bpe")
        elif not states and opt["embedding_type"] != "random":
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(model.decoder.lt.weight, opt["embedding_type"])
            if opt["lookuptable"] in ["unique", "dec_out"]:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(
                    model.encoder.lt.weight, opt["embedding_type"], log=False
                )

        if states:
            # set loaded states if applicable
            model.load_state_dict(states["model"])

        if opt["embedding_type"].endswith("fixed"):
            print("Seq2seq: fixing embedding weights.")
            model.decoder.lt.weight.requires_grad = False
            model.encoder.lt.weight.requires_grad = False
            if opt["lookuptable"] in ["dec_out", "all"]:
                model.output.weight.requires_grad = False

        return model

    def batchify(self, obs_batch, sort=True):
        """
        Add action and attribute supervision for batches.

        Store history vec as context_vec.
        """
        # NOTE: `sort` is set to True here (Default is False in TorchGeneratorAgent)
        # TODO: Sorting the batch will result in various local metrics being broadcasted
        # back to individual examples in the wrong order, such as the lengths of
        # the context and labels. Aggregate metric reports will still be accurate.
        batch = super().batchify(obs_batch, sort)
        # sum here is list concat, not addition
        context_vec, hist_lens_ = self._pad_tensor(
            sum([obs_batch[i]['context_vec'] for i in batch.valid_indices], [])
        )
        batch['context_vec'] = context_vec
        batch['hist_lens'] = torch.LongTensor(
            [len(obs_batch[i]['context_vec']) for i in batch.valid_indices]
        )
        return batch

    def _model_input(self, batch):
        return (batch.text_vec, batch.context_vec, batch.hist_lens)

    def _set_text_vec(self, obs, history, truncate):
        """
        Set the 'text_vec' field in the observation.

        Overridden to include both local utterance (text_vec) and full history
        (context_vec)
        """
        if "text" not in obs:
            return obs

        if "text_vec" not in obs:
            # text vec is not precomputed, so we set it using the history
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs
            obs["full_text"] = history_string
            if history_string:
                history_vec = history.get_history_vec_list()
                obs["text_vec"] = history_vec[-1]
                obs["full_text_vec"] = history.get_history_vec()
                obs["context_vec"] = history_vec

        # check truncation
        if obs.get("text_vec") is not None:
            truncated_vec = self._check_truncate(obs["text_vec"], truncate, True)
            obs.force_set("text_vec", torch.LongTensor(truncated_vec))
        return obs
