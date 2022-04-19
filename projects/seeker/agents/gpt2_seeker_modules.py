#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
GPT2 SeeKeR Agent for Language Modeling.
"""
from typing import Optional, Type, Tuple, List, Dict, Any
import torch
import torch.nn

from parlai.agents.fid.fid import FidModel, concat_enc_outs
from parlai.agents.hugging_face.gpt2 import GPT2Decoder, Gpt2Agent
from parlai.agents.rag.args import RetrieverType
from parlai.agents.rag.modules import RagModel
from parlai.agents.rag.retrievers import (
    Document,
    BLANK_DOC,
    SearchQuerySearchEngineRetriever,
    retriever_factory,
)
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
import parlai.utils.logging as logging

from projects.seeker.agents.seeker_modules import (
    ComboFidSearchQuerySearchEngineRetriever,
)
from projects.seeker.agents.seeker_modules import (
    combo_fid_retriever_factory,
    interleave_fid_combo_outputs,
    ComboFidModel,
)

########################
# Retrieval Components #
########################


class FilterDocsForLabelSearchEngineRetrieverMixin:
    """
    Mixin for filtering the document containing the label out from retrieved docs.

    As these models are generally used with a CC knowledge dump, and the task on which
    they are evaluated is from the same CC dump, it is useful to filter out the document
    containing the label string from the retrieved documents, for a fair comparison.
    """

    def set_labels(self, labels: torch.Tensor):
        """
        Cache the label vec.
        """
        self.label_vec = labels

    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Override retrieve and score to filter out docs that contain the label string.

        Copy over the whole thing because we need to check before chunking.
        """
        # step 1
        search_queries = self.generate_search_query(query)  # type: ignore

        # step 2
        search_results_batch = self.search_client.retrieve(
            search_queries, self.n_docs
        )  # type: ignore

        # step 3
        top_docs = []
        top_doc_scores = []
        max_n_docs: int = self.n_docs  # type: ignore
        for batch_id, (sq, search_results) in enumerate(
            zip(search_queries, search_results_batch)
        ):
            if not search_results:
                search_results = self._empty_docs(self.n_docs)  # type: ignore
            elif len(search_results) < self.n_docs:  # type: ignore
                remain_docs = self.n_docs - len(search_results)  # type: ignore
                search_results.extend(self._empty_docs(remain_docs))  # type: ignore
            docs_i = []
            scors_i = []
            # Change this debug later
            logging.debug(
                f'URLS:\n{self._display_urls(search_results)}'
            )  # type: ignore
            label_text = self.dict.vec2txt(
                self.label_vec[batch_id, :-1]
            )  # type: ignore
            for i, doc in enumerate(search_results):
                url = doc['url']
                title = doc['title']
                dcontent = doc['content']
                assert type(dcontent) in (
                    str,
                    list,
                ), f'Unrecognized retrieved doc: {dcontent}'
                full_text = (
                    dcontent if isinstance(dcontent, str) else '\n'.join(doc['content'])
                )
                if label_text in full_text:
                    docs_i.append(BLANK_DOC)
                    scors_i.append(0)
                else:
                    doc_chunks = [
                        dc[0]
                        for dc in self.pick_chunk(
                            sq, title, full_text, url
                        )  # type: ignore
                    ]
                    for splt_id, splt_content in enumerate(doc_chunks):
                        docs_i.append(
                            Document(
                                docid=url, text=splt_content, title=f'{title}_{splt_id}'
                            )
                        )
                        scors_i.append(self.rank_score(i))  # type: ignore
            max_n_docs = max(max_n_docs, len(docs_i))
            top_docs.append(docs_i)
            top_doc_scores.append(scors_i)
        # Pad with empty docs
        for i in range(len(top_docs)):
            n_empty = max_n_docs - len(top_docs[i])
            if n_empty:
                top_docs[i] = top_docs[i] + [BLANK_DOC] * n_empty
                top_doc_scores[i] = top_doc_scores[i] + [0] * n_empty
        self.top_docs = top_docs
        self.search_queries = search_queries
        return top_docs, torch.Tensor(top_doc_scores).to(query.device)


class FilterDocsForLabelSearchEngineRetrieverCombo(
    FilterDocsForLabelSearchEngineRetrieverMixin,
    ComboFidSearchQuerySearchEngineRetriever,
):
    pass


class FilterDocsForLabelSearchEngineRetrieverBase(
    FilterDocsForLabelSearchEngineRetrieverMixin, SearchQuerySearchEngineRetriever
):
    pass


#######################################
# Special Modules/Agents for GPT2 FiD #
#######################################


class IdentityLayer(torch.nn.Module):
    """
    Acts as the encoder for the FiD model.

    Custom Identity layer (as opposed to parlai.utils.torch.IdentityLayer) to account
    for special output required.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, *args, **kwargs):
        super().__init__()
        self.pad_idx = dictionary[dictionary.null_token]

    def forward(self, input: torch.LongTensor, *args, **kwargs):
        return input, input.ne(self.pad_idx)


class ComboGPT2Decoder(GPT2Decoder):
    """
    Allows us to use standard FiD/RAG model building logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args[:2])


#############################
# GPT2 With Retriever Model #
#############################


class GPT2WithRetrieverModel(FidModel):
    """
    GPT2 with Retriever Model.

    A GPT2 model that receives documents from a retriever.
    """

    def __init__(self, opt, dictionary, retriever_shared=None):
        self.add_start_token = opt["add_start_token"]
        opt['converting'] = True  # set not to build the retriever
        FidModel.__init__(self, opt, dictionary, retriever_shared)
        opt['converting'] = False
        self.config = self.seq2seq_decoder.transformer.config
        self.embedding_size = self.config.n_embd
        self.lm_head = torch.nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False
        )
        self._tie_weights(self.lm_head, self.seq2seq_decoder.transformer.wte)
        self.doc_delim = self.dict.txt2vec(Document.PASSAGE_DELIM)[0]
        self.min_doc_len = opt['min_doc_token_length']
        self.truncate = (
            opt['text_truncate'] if opt['text_truncate'] > -1 else opt['truncate']
        )
        if opt.get('filter_docs_with_label'):
            assert (
                RetrieverType(opt['rag_retriever_type']) == RetrieverType.SEARCH_ENGINE
            )
            self.retriever = FilterDocsForLabelSearchEngineRetrieverBase(
                opt, dictionary, shared=retriever_shared
            )  # type: ignore
        else:
            self.retriever = retriever_factory(opt, dictionary, shared=retriever_shared)

    @classmethod
    def build_encoder(
        cls,
        opt: Opt,
        *args,
        dictionary: Optional[DictionaryAgent] = None,
        embedding: Optional[torch.nn.Embedding] = None,
        encoder_class: Optional[Type] = None,
        **kwargs,
    ):
        """
        Override to build with IdentityLayer as Encoder.
        """
        return FidModel.build_encoder(
            opt, dictionary, encoder_class=IdentityLayer, **kwargs
        )

    @classmethod
    def build_decoder(
        cls,
        opt: Opt,
        *args,
        embedding: Optional[torch.nn.Embedding] = None,
        n_positions: Optional[int] = None,
        decoder_class: Optional[Type] = None,
        **kwargs,
    ):
        """
        Override to use GPT2 as decoder.
        """
        return FidModel.build_decoder(
            opt,
            Gpt2Agent.dictionary_class()(opt),
            decoder_class=ComboGPT2Decoder,
            **kwargs,
        )

    def encoder(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        query_vec: torch.LongTensor,
        input_turns_cnt: torch.LongTensor,
        target_lengths: Optional[torch.LongTensor],
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
        Override FidModel.encoder to pack all the documents into one input example.

        :param input:
            2D [bsz, seqlen] input to the encoder
        :param input_lengths:
            1D [bsz] lengths of each input item
        :param query_vec:
            2D [bsz*n_turns, seqlen] input for the retriever
        :param input_turns_cnt:
            1D [bsz] number of dialogue turns for each input example
        :param input_lengths:
            1D [bsz] lengths of each target item (for each input item)

        :return (encoder_out, encoder_mask, input_turns_cnt, top_docs, top_doc_scores):
            encoder_out: *concatenated* encoded representations of context/document pairs
            encoder_mask: new mask for enc_out
            input_turns_cnt: pass along the input turns count for the decoder
            top_docs: List of top Documents for each batch example
            top_doc_scores: scores for each retrieved document.
        """
        enc_out, mask, input_turns_cnt, top_docs, top_doc_scores = RagModel.encoder(
            self, input, input_lengths, query_vec, input_turns_cnt, positions, segments
        )  # type: ignore
        seq_len, n_docs = enc_out.size(1), enc_out.size(0) // input.size(0)

        if input_turns_cnt is not None:
            # Input Turns is a tensor of dim [bsz]
            input = input.repeat_interleave(input_turns_cnt, dim=0)  # type: ignore
        doc_starts = (enc_out == self.pad_idx).sum(dim=1)  # left padded
        doc_lens = (seq_len - input_lengths.repeat_interleave(n_docs)) - doc_starts
        # if no padding, docs are assumed to be min doc length long
        doc_lens[doc_lens.le(0)] = self.min_doc_len
        new_enc_out = enc_out.clone()
        # BEFORE:
        #  [
        #   pad...doc_0 / in_0
        #   pad...doc_1 / in_0
        #   ...
        #   pad...doc_n / in_m
        #                       ]
        total_length_i = 0
        for i, doc_len in enumerate(doc_lens):
            if i % n_docs == 0:
                total_length_i = 0
            # max doc length is determined by how much space we have after subtracting input length
            input_and_target = input_lengths[i // n_docs] + (
                target_lengths[i // n_docs] if target_lengths is not None else 0
            )
            max_doc_len = torch.div(
                (seq_len - input_and_target), n_docs, rounding_mode='floor'
            )
            max_doc_len = max(max_doc_len, self.min_doc_len)  # type: ignore
            doc_len = min(doc_len, max_doc_len)
            total_length_i += doc_len
            if i % n_docs == n_docs - 1:
                # keep the actual input context when processing the last doc.
                clamped_input_length = input_lengths[i // n_docs].clamp(
                    max=self.truncate - total_length_i
                )
                if target_lengths is not None:
                    clamped_input_length = input_lengths[i // n_docs].clamp(
                        max=self.truncate - total_length_i - target_lengths
                    )

                pad_end = seq_len - clamped_input_length - doc_len
            else:
                pad_end = seq_len - doc_len
            # we simply move the portion of the doc we want to keep to the end of the tensor,
            # and mask out everything else.
            mask[i, :pad_end] = False
            new_enc_out[i, pad_end : pad_end + doc_len] = enc_out[
                i, doc_starts[i] : doc_starts[i] + doc_len
            ]

        new_out, new_mask = concat_enc_outs(
            input, new_enc_out.unsqueeze(-1), mask, 1, self.pad_idx, right_padded=False
        )
        # After:
        #  [
        #   doc_0 / doc_1 / doc_2 ... in_0
        #   doc_0 / doc_1 / doc_2 ... in_m
        #                                   ]
        return new_out, new_mask, input_turns_cnt, top_docs, top_doc_scores

    def decoder(
        self,
        input: torch.LongTensor,
        encoder_state: Tuple[Any, ...],
        incr_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Override to make sure dimensions of enc_out are appropriate.

        :param input:
            input for the decoder
        :param encoder_state:
            RAG encoder states
        :param incr_state:
            incremental decoder state

        :return (output, new_incr_state):
            return the output token distribution, as well as new incremental state.
        """
        enc_out, *_ = encoder_state
        if enc_out.dim() == 2:
            enc_out = enc_out.t()
        else:
            enc_out = enc_out.squeeze(-1)
        dec_out, incr_state = self.seq2seq_decoder(
            input, enc_out, incr_state
        )  # type: ignore
        return dec_out, incr_state

    def concat_docs_and_input(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        top_docs: List[List[Document]],
        max_num_docs: int,
        right_padded: bool = True,
    ) -> torch.LongTensor:
        """
        Override to account for left-padded input.
        """
        return super().concat_docs_and_input(
            input, input_lengths, top_docs, max_num_docs, right_padded=False
        )

    def set_labels(self, labels: torch.LongTensor):
        """
        Set retriever's labels.
        """
        assert self.retriever is not None
        self.retriever.set_labels(labels)

    ### HFGPT2Model Functions

    def _get_decoder(self, opt, dict):
        return GPT2Decoder(opt, dict)

    def _tie_weights(self, output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.weight

    def _get_special_tokens(self, opt, dict):
        return dict.null_idx, dict.start_idx, dict.end_idx

    def output(self, tensor):
        """
        Compute output logits.
        """
        return self.lm_head(tensor)

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        new_incr_state = []
        for layer_past in incremental_state:
            if torch.is_tensor(layer_past):
                new_incr_state.append(torch.index_select(layer_past, 1, inds))
            else:
                # newer versions of HF split up the intermediate outputs
                assert isinstance(layer_past, tuple)
                layer_past = torch.stack(layer_past, dim=0)
                new_incr_state.append(torch.index_select(layer_past, 1, inds))

        return tuple(new_incr_state)

    def decode_forced(self, encoder_states, ys):
        """
        Override to get rid of start token input.
        """
        if self.add_start_token:
            return super().decode_forced(encoder_states, ys)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds


####################
# Combo GPT2 Model #
####################


class ComboGPT2Model(GPT2WithRetrieverModel, ComboFidModel):
    """
    Combo model.

    This model can handle contexts with and without retrieved documents.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, retriever_shared=None):
        super().__init__(opt, dictionary, retriever_shared)
        if opt.get('filter_docs_with_label'):
            assert (
                RetrieverType(opt['rag_retriever_type']) == RetrieverType.SEARCH_ENGINE
            )
            self.retriever = FilterDocsForLabelSearchEngineRetrieverCombo(
                opt, dictionary, shared=retriever_shared
            )  # type: ignore
        else:
            self.retriever = combo_fid_retriever_factory(
                opt, dictionary, shared=retriever_shared
            )
        self.top_docs: List[List[Document]] = []

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
        target_lengths: torch.LongTensor,
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
        Copy most of the ComboFidModel.forward here, but include the target_lengths for
        the encoder.

        Note, however, that we don't use it.
        """
        # If we have a full batch of retrieve (or not), can use simple logic here.
        if torch.all(skip_retrieval_vec):
            new_out, new_mask = self.seq2seq_encoder(input, positions, segments)
            return (
                new_out.unsqueeze(-1),
                new_mask,
                None,
                None,
                None,
            )  # need to unsqueeze to handle in decoder
        elif torch.all(~skip_retrieval_vec):
            output = super().encoder(
                input,
                input_lengths,
                query_vec,
                input_turns_cnt,
                None,
                positions,
                segments,
            )
            self.top_docs = output[-2]  # type: ignore
            return output
        assert all(t is None for t in [input_turns_cnt, positions, segments])
        # Encode with `super()` call for non-skip-retrieval inputs
        (
            enc_out_retrieval,
            mask_retrieval,
            _,
            top_docs,
            top_doc_scores,
        ) = super().encoder(
            input[~skip_retrieval_vec],  # type: ignore
            input_lengths[~skip_retrieval_vec],  # type: ignore
            query_vec[~skip_retrieval_vec],  # type: ignore
            input_turns_cnt,
            None,  # don't pass target lengths when combining
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
            enc_out_skip_retrieval.unsqueeze(-1),
            mask_retrieval,
            mask_skip_retrieval,
            skip_retrieval_vec,
            top_docs,  # type: ignore
            top_doc_scores,  # type: ignore
            right_padded=False,
        )
        self.top_docs = new_top_docs
        return new_out, new_mask, input_turns_cnt, new_top_docs, new_top_doc_scores
