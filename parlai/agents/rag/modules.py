#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from typing import List, Optional, Type, Tuple

from parlai.agents.rag.classes import Document
from parlai.agents.rag.interfaces import RAG, RagRetriever
from parlai.agents.transformer.interfaces import Transformer
from parlai.agents.transformer.modules import TransformerGeneratorModel
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.utils.torch import padded_tensor, FP16_PAD_SIZE


TITLE_DELIM = ' / '
PASSAGE_DELIM = ' // '


class QueryModelType(Enum):
    BERT = 'bert'
    BERT_FROM_PARLAI_RAG = 'bert_from_parlai_rag'


class DefaultRagModel(nn.Module, RAG):
    @dataclass
    class Manifest(RAG.Manifest):
        retriever: Type[RagRetriever]
        retriever_manifest: RagRetriever.Manifest
        generator: Type[Transformer] = TransformerGeneratorModel
        generator_manifest: Transformer.Manifest = TransformerGeneratorModel.Manifest()

    def __init__(
        self, opt: Opt, dictionary: DictionaryAgent, manifest: Manifest = None, **kwargs
    ):
        super().__init__()
        manifest = manifest or type(self).Manifest()
        # attrs
        self.rag_model_type = opt['rag_model_type']
        self.generation_model = opt['generation_model']
        # modules
        self.retriever = manifest.retriever(
            opt=opt, manifest=manifest.retriever_manifest
        )
        self.generator = manifest.generator(
            opt=opt, dictionary=dictionary, manifest=manifest.generator_manifest
        )

    def forward(
        self,
        input: torch.LongTensor,
        input_lengths: List[int],
        input_text: Optional[List[str]],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        top_docs, _ = self.retriever.retrieve_and_score(input_text)
        expanded_input, expanded_input_lengths = self._add_docs_to_input(
            input, input_lengths, top_docs
        )
        return self.generator(expanded_input)

    def _add_docs_to_input(
        self,
        input: torch.LongTensor,
        input_lengths: List[int],
        top_docs: List[List[Document]],
    ) -> Tuple[torch.LongTensor, List[int]]:
        """
        Add document tokens to input tokens.

        :param input:
            original input tokens
        :param input_lengths:
            original input lengths
        :param top_docs:
            list of n_docs top documents for each input sequence

        :return (tokens, lengths):
            return expanded token vectors & corresponding lengths
        """
        max_len = self.expanded_input_truncate
        expanded_input: List[torch.LongTensor] = []
        max_num_docs = max([len(d) for d in top_docs])
        for i, docs in enumerate(top_docs):
            for rank in range(len(docs)):
                input_i = input[i, :]
                doc = docs[rank]
                passage_str = f"{doc.get_title().strip()}{TITLE_DELIM}{doc.get_text().strip()}{PASSAGE_DELIM}"
                if self.variant == 'bart' and self.n_extra_positions <= 0:
                    # move SOS to start of passage since we append question to end
                    input_i = input_i[1:]
                    sample_doc_tokens = torch.LongTensor(
                        [self.dict[self.dict.start_token]]
                        + self.dict.txt2vec(passage_str)
                    ).to(input)
                else:
                    sample_doc_tokens = torch.LongTensor(
                        self.dict.txt2vec(passage_str)
                    ).to(input)

                if self.n_extra_positions <= 0:
                    # Prepend document to text
                    input_i_len = input_lengths[i]
                    new_input_length = min(
                        self.expanded_input_truncate - self.min_doc_token_length,
                        input_i_len,
                    )
                    input_i = input_i[input_i_len - new_input_length : input_i_len]
                    doc_max_len = max(max_len - len(input_i), 0)
                    sample_doc_tokens = sample_doc_tokens[:doc_max_len]
                    expanded_input.append(
                        torch.cat([sample_doc_tokens, input_i])[:max_len]
                    )
                else:
                    # Append Document to text
                    sample_doc_tokens = sample_doc_tokens[:max_len]
                    input_i_new = input_i.new(self.n_positions - self.n_extra_positions)
                    input_i_new.fill_(self.dict[self.dict.null_token])
                    input_i_new[: input_i.size(0)] = input_i
                    expanded_input.append(torch.cat([input_i_new, sample_doc_tokens]))
            if self.include_null_doc:
                expanded_input.append(
                    input[i, :]
                )  # use original, in case bart w/ start tokens
            # append extra null inputs if there are diff # of docs per input
            expanded_input += [
                input_i.new(input_i.size()).fill_(self.dict[self.dict.null_token])
            ] * (max_num_docs - len(docs))
        input, input_lengths = padded_tensor(
            expanded_input,
            use_cuda=input.is_cuda,
            fp16friendly=input.size(1) % FP16_PAD_SIZE == 0,
            max_len=max_len if self.n_extra_positions <= 0 else None,
            pad_idx=self.dict[self.dict.null_token],
            device=input.device.index,
        )
        return input, input_lengths
