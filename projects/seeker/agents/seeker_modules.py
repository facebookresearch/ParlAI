#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Modules for SeeKeR.
"""
import torch
from typing import List, Tuple, Optional

from parlai.agents.fid.fid import FidModel
from parlai.agents.rag.args import RetrieverType
from parlai.agents.rag.retrievers import (
    Document,
    BLANK_DOC,
    SearchQuerySearchEngineRetriever,
    RagRetriever,
    retriever_factory,
)
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
import parlai.utils.logging as logging


def interleave_fid_combo_outputs(
    enc_out_retrieval: torch.Tensor,
    enc_out_skip_retrieval: torch.Tensor,
    mask_retrieval: torch.Tensor,
    mask_skip_retrieval: torch.Tensor,
    skip_retrieval_vec: torch.Tensor,
    top_docs: List[List[Document]],
    top_doc_scores: torch.Tensor,
    right_padded: bool = True,
) -> Tuple[torch.Tensor, torch.BoolTensor, List[List[Document]], torch.Tensor]:
    """
    Interleave FiD encoder outputs.

    Outputs are either encodings of documents/input,
    or encodings of just input.

    This operation is to preserve original batch order.

    :param enc_out_retrieval:
        encoder output for inputs with retrieved documents
    :param enc_out_skip_retrieval:
        encoder output for inputs without retrieved documents
    :param mask_retrieval:
        mask for inputs with retrieved documents
    :param mask_skip_retrieval:
        mask for inputs without retrieved documents
    :param skip_retrieval_vec:
        1D tensor indicating which inputs skip retrieval vs. dont
    :param top_docs:
        top documents for outputs with retrieved documents
    :param top_doc_scores:
        top doc scores.
    :param right_padded:
        whether the final output is right padded

    :return (new_out, new_mask, new_top_docs, new_top_doc_scores):
        return interleaved concatenation of inputs
    """
    bsz = enc_out_retrieval.size(0) + enc_out_skip_retrieval.size(0)
    dim = enc_out_retrieval.size(-1)
    seqlen = max(enc_out_retrieval.size(1), enc_out_skip_retrieval.size(1))
    n_docs = top_doc_scores.size(1)

    # The following logic interleaves the results in new tensors.
    new_out = enc_out_retrieval.new(bsz, seqlen, dim).fill_(0)
    new_mask = mask_retrieval.new(bsz, seqlen).fill_(False)
    new_top_doc_scores = top_doc_scores.new(bsz, n_docs)
    new_top_docs = []
    retr_offset = 0
    skip_offset = 0
    for i, skip in enumerate(skip_retrieval_vec):
        if skip:
            vec = enc_out_skip_retrieval[skip_offset]
            mask = mask_skip_retrieval[skip_offset]
            scores = torch.zeros(n_docs).to(top_doc_scores)
            docs = [BLANK_DOC] * n_docs
            skip_offset += 1
        else:
            vec = enc_out_retrieval[retr_offset]
            mask = mask_retrieval[retr_offset]
            scores = top_doc_scores[retr_offset]
            docs = top_docs[retr_offset]
            retr_offset += 1
        if right_padded:
            new_out[i, : vec.size(0), :] = vec
            new_mask[i, : mask.size(0)] = mask
        else:
            new_out[i, -vec.size(0) :, :] = vec
            new_mask[i, -mask.size(0) :] = mask

        new_top_doc_scores[i] = scores
        new_top_docs.append(docs)

    return new_out, new_mask, new_top_docs, new_top_doc_scores  # type: ignore


class ComboFidSearchRetrieverMixin:
    """
    Mixin for Combo Search Modules.

    These models **either** search **or** do not.
    """

    def set_search_queries(self, queries: List[str]):
        """
        Bypass SQ generation by setting the search queries.
        """
        self.search_queries = queries

    def init_search_query_generator(self, opt):
        """
        If a generator model file is specified, we use it.
        """
        if opt['search_query_generator_model_file']:
            return super().init_search_query_generator(opt)  # type: ignore
        else:
            logging.warning('Not initializing SQ Generator *within* retriever')

    def generate_search_query(self, query: torch.LongTensor) -> List[str]:
        """
        Generate search queries.

        If there are cached queries, use those; otherwise, generate our own.
        """
        if self.search_queries:
            # Ensure we only search for examples with well-formed queries
            # as this list is bsz-length, not search-bsz-length.
            search_queries = [q for q in self.search_queries if q]
        else:
            search_queries = super().generate_search_query(query)  # type: ignore
        return search_queries


class ComboFidSearchQuerySearchEngineRetriever(
    ComboFidSearchRetrieverMixin, SearchQuerySearchEngineRetriever
):
    """
    Override Search Engine Retriever to accommodate SQ Generator from Combo Setup.
    """


def combo_fid_retriever_factory(
    opt: Opt, dictionary: DictionaryAgent, shared=None
) -> Optional[RagRetriever]:
    """
    Bypass call to standard retriever factory to possibly build our own retriever.
    """
    if opt.get('converting'):
        return None
    retriever = RetrieverType(opt['rag_retriever_type'])
    if retriever is RetrieverType.SEARCH_ENGINE:
        return ComboFidSearchQuerySearchEngineRetriever(
            opt, dictionary, shared=shared
        )  # type: ignore
    else:
        return retriever_factory(opt, dictionary, shared)


class ComboFidModel(FidModel):
    """
    Combo FiD Model.

    This model receives a new input that indicates which batch items require
    retrieval/FiD encoding, and which can be used in a standard encoder/decoder pass.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, retriever_shared=None):
        super().__init__(opt, dictionary, retriever_shared)
        self.retriever = combo_fid_retriever_factory(
            opt, dictionary, shared=retriever_shared
        )
        self.top_docs = []

    def set_search_queries(self, queries: List[str]):
        """
        Set retriever's search queries.
        """
        assert self.retriever is not None
        self.retriever.set_search_queries(queries)

    def get_top_docs(self) -> List[List[Document]]:
        """
        Return Documents.
        """
        return self.top_docs

    def encoder(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        query_vec: torch.LongTensor,
        input_turns_cnt: torch.LongTensor,
        skip_retrieval_vec: torch.BoolTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.BoolTensor,
        Optional[torch.LongTensor],
        Optional[List[List[Document]]],
        Optional[torch.Tensor],
    ]:
        """
        Override Standard FiD.encoder to account for skip_retrieval inputs.

        With `skip_retrieval` inputs, we forward directly through the `seq2seq_encoder`

        Otherwise, we call super().encoder to retrieve and concatenate outputs.

        :param input:
            2D [bsz, seqlen] input to the encoder
        :param input_lengths:
            1D [bsz] lengths of each input item
        :param query_vec:
            2D [bsz*n_turns, seqlen] input for the retriever
        :param input_turns_cnt:
            1D [bsz] number of dialogue turns for each input example
        :param skip_retrieval_vec:
            1D [bsz] indicator for whether to skip retrieval.

        :return (encoder_out, encoder_mask, input_turns_cnt, top_docs, top_doc_scores):
            encoder_out: *concatenated* (or not) encoded representations of context/document pairs
            encoder_mask: new mask for enc_out
            input_turns_cnt: pass along the input turns count for the decoder
            top_docs: List of top Documents for each batch example
            top_doc_scores: scores for each retrieved document.
        """
        # If we have a full batch of retrieve (or not), can use simple logic here.
        if torch.all(skip_retrieval_vec):
            new_out, new_mask = self.seq2seq_encoder(input, positions, segments)
            return new_out, new_mask, None, None, None
        elif torch.all(~skip_retrieval_vec):
            output = super().encoder(
                input, input_lengths, query_vec, input_turns_cnt, positions, segments
            )
            self.top_docs = output[-2]
            return output

        assert all(t is None for t in [input_turns_cnt, positions, segments])
        # Encode with `super()` call for non-skip-retrieval inputs
        (
            enc_out_retrieval,
            mask_retrieval,
            input_turns_cnt,
            top_docs,
            top_doc_scores,
        ) = super(ComboFidModel, self).encoder(
            input[~skip_retrieval_vec],
            input_lengths[~skip_retrieval_vec],
            query_vec[~skip_retrieval_vec],
            input_turns_cnt,
            positions,
            segments,
        )
        # Encode with seq2seq_encoder for skip-retrieval inputs
        enc_out_skip_retrieval, mask_skip_retrieval = self.seq2seq_encoder(
            input[skip_retrieval_vec]
        )

        (
            new_out,
            new_mask,
            new_top_docs,
            new_top_doc_scores,
        ) = interleave_fid_combo_outputs(
            enc_out_retrieval,
            enc_out_skip_retrieval,
            mask_retrieval,
            mask_skip_retrieval,
            skip_retrieval_vec,
            top_docs,
            top_doc_scores,
        )
        self.top_docs = new_top_docs

        return new_out, new_mask, input_turns_cnt, new_top_docs, new_top_doc_scores
