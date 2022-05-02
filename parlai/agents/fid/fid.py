#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering.

See https://arxiv.org/abs/2007.01282
"""
from abc import abstractmethod
from copy import deepcopy
import torch
import random
from typing import Tuple, Union, Optional, List, Dict, Any

from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.agents.transformer.transformer import TransformerGeneratorModel

from parlai.agents.rag.args import RetrieverType
from parlai.agents.rag.modules import RagModel, Document, T5RagModel
from parlai.agents.rag.rag import RagAgent
from parlai.agents.rag.model_types import (
    RagToken,
    get_forced_decoder_inputs,
    fix_incremental_state,
)
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
from parlai.tasks.wizard_of_internet import constants as consts


class Fid(RagToken):
    """
    FiD mimics RAG Token interface in many ways; we simply need to adjust the decoder
    inputs to not repeat, as FiD attends over all encoder outputs jointly.
    """

    def get_initial_forced_decoder_input(
        self,
        bsz: int,
        inputs: torch.LongTensor,
        n_docs: int,
        start_idx: int,
        end_idx: int,
        input_turns_cnt: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return get_forced_decoder_inputs(
            inputs, bsz, start_idx, end_idx, self.generation_model
        )

    def get_initial_decoder_input(self, input: torch.LongTensor) -> torch.LongTensor:
        return input

    def get_next_decoder_input(
        self,
        prev_input: torch.LongTensor,
        selection: torch.LongTensor,
        incr_state_inds: torch.LongTensor,
    ) -> torch.LongTensor:
        prev_input = torch.index_select(prev_input, 0, incr_state_inds)  # type: ignore
        decoder_input = torch.cat([prev_input, selection], dim=-1)
        return decoder_input  # type: ignore


class FidModel(RagModel):
    """
    The FiD Model is a simpler version of the RAG Model.

    We override the encoder and decoder methods to join encoder outputs, and decode
    normally, respectively.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, retriever_shared=None):
        super().__init__(opt, dictionary, retriever_shared=retriever_shared)
        self._rag_model_interface = Fid(opt, dictionary[dictionary.null_token])
        self.embedding_size = opt['embedding_size']

    def reorder_encoder_states(
        self,
        encoder_states: Tuple[torch.Tensor, ...],
        indices: Union[List[int], torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[Document]], torch.Tensor]:
        """
        Override RagModel.reorder_encoder_states to make sure we only pass enc, mask.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask, *_ = encoder_states
        return TransformerGeneratorModel.reorder_encoder_states(
            self, (enc, mask), indices
        )

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Override RagModel.reorder_decoder_incremental_state to resort back to normal
        reordering.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.
        """
        incremental_state = fix_incremental_state(
            self.generation_model, incremental_state
        )
        if not incremental_state:
            return incremental_state
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.seq2seq_decoder.layers)
        }

    def encoder(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        query_vec: torch.LongTensor,
        input_turns_cnt: torch.LongTensor,
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
        Concatenate all encoder outputs in model forward.

        :param input:
            2D [bsz, seqlen] input to the encoder
        :param input_lengths:
            1D [bsz] lengths of each input item
        :param query_vec:
            2D [bsz*n_turns, seqlen] input for the retriever
        :param input_turns_cnt:
            1D [bsz] number of dialogue turns for each input example

        :return (encoder_out, encoder_mask, input_turns_cnt, top_docs, top_doc_scores):
            encoder_out: *concatenated* encoded representations of context/document pairs
            encoder_mask: new mask for enc_out
            input_turns_cnt: pass along the input turns count for the decoder
            top_docs: List of top Documents for each batch example
            top_doc_scores: scores for each retrieved document.
        """
        enc_out, mask, input_turns_cnt, top_docs, top_doc_scores = super().encoder(
            input, input_lengths, query_vec, input_turns_cnt, positions, segments
        )  # type: ignore

        if input_turns_cnt is not None:
            # Input Turns is a tensor of dim [bsz]
            input = input.repeat_interleave(input_turns_cnt, dim=0)  # type: ignore

        new_out, new_mask = concat_enc_outs(
            input, enc_out, mask, self.embedding_size, self.pad_idx
        )

        return new_out, new_mask, input_turns_cnt, top_docs, top_doc_scores

    def decoder(
        self,
        input: torch.LongTensor,
        encoder_state: Tuple[Any, ...],
        incr_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Decode, RAG-Style.

        :param input:
            input for the decoder
        :param encoder_state:
            RAG encoder states
        :param incr_state:
            incremental decoder state

        :return (output, new_incr_state):
            return the output token distribution, as well as new incremental state.
        """
        enc_out, enc_mask, *_ = encoder_state
        dec_out, incr_state = self.seq2seq_decoder(
            input, (enc_out, enc_mask), incr_state
        )  # type: ignore
        dec_out = self.decoder_output(dec_out)
        return dec_out, incr_state


class T5FidModel(FidModel, T5RagModel):
    def __init__(self, opt: Opt, dictionary: DictionaryAgent, retriever_shared=None):
        super().__init__(opt, dictionary, retriever_shared=retriever_shared)
        self.embedding_size = self.t5.model_dim


class FidAgent(RagAgent):
    """
    Fusion in Decoder Agent.

    Fusion in Decoder is very similar to RAG; each requires a retrieval and subsequent
    generation step.

    The difference is that FiD will encode all documents in parallel in encoder,
    concatenate, and feed as one giant encoding to Decoder.

    This forces the Decoder to attend over the several documents directly,
    rather than marginalizing later.

    As such, FidAgent is a natural extension of the RagAgent. I've extracted out to its
    own agent for ease of use.
    """

    @property
    def rag_model_type(self) -> str:
        return self._rag_model_type

    @rag_model_type.setter
    def rag_model_type(self, model: str):
        self._rag_model_type = model
        self._rag_model_interface = Fid(self.opt, self.NULL_IDX)

    def build_model(self) -> FidModel:
        if self.generation_model == 't5':
            model = T5FidModel(self.opt, self.dict)
        else:
            model = FidModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model


RETRIEVER_DOC_LEN_TOKENS = 256


class SearchQueryFiDAgent(FidAgent):
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt=None):
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('Search Query FiD Params')

        # Search Query generator
        group.add_argument(
            '--search-query-generator-model-file',
            type=str,
            help='Path to a query generator model.',
        )
        group.add_argument(
            '--search-query-generator-inference',
            type=str,
            default='greedy',
            help='Generation algorithm for the search query generator model',
        )
        group.add_argument(
            '--search-query-generator-beam-min-length',
            type=int,
            default=1,
            help='The beam_min_length opt for the search query generator model',
        )
        group.add_argument(
            '--search-query-generator-beam-size',
            type=int,
            default=1,
            help='The beam_size opt for the search query generator model',
        )
        group.add_argument(
            '--search-query-generator-text-truncate',
            type=int,
            default=512,
            help='Truncates the input to the search query generator model',
        )

        # Creating chunks and spliting the documents
        group.add_argument(
            '--splitted-chunk-length',
            type=int,
            default=RETRIEVER_DOC_LEN_TOKENS,
            help='The number of tokens in each document split',
        )
        group.add_argument(
            '--doc-chunk-split-mode',
            type=str,
            choices=['word', 'token'],
            default='word',
            help='split the docs by white space (word) or dict tokens.',
        )
        group.add_argument(
            '--n-ranked-doc-chunks',
            type=int,
            default=1,
            help='Number of document chunks to keep if documents is too long and has to be splitted.',
        )
        group.add_argument(
            '--doc-chunks-ranker',
            type=str,
            choices=['tfidf', 'head', 'woi_chunk_retrieved_docs'],
            default='head',
            help='How to rank doc chunks.',
        )
        parser.add_argument(
            '--woi-doc-chunk-size',
            default=500,
            type=int,
            help='Document chunk size (in characters).',
        )

        return parser


class SearchQuerySearchEngineFiDAgent(SearchQueryFiDAgent):
    def __init__(self, opt: Opt, shared: TShared = None):
        opt = deepcopy(opt)
        opt['rag_retriever_type'] = RetrieverType.SEARCH_ENGINE.value
        super().__init__(opt, shared=shared)

    @classmethod
    def add_cmdline_args(cls, parser, partial_opt=None):
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('Search Engine FiD Params')
        group.add_argument('--search-server', type=str, help='A search server address.')
        return parser


class SearchQueryFAISSIndexFiDAgent(SearchQueryFiDAgent):
    def __init__(self, opt: Opt, shared: TShared = None):
        opt = deepcopy(opt)
        opt['rag_retriever_type'] = RetrieverType.SEARCH_TERM_FAISS.value
        super().__init__(opt, shared=shared)


class GoldDocRetrieverFiDAgent(SearchQueryFiDAgent):
    """
    Uses the gold retrieved docs (documents shown to crowdsourcing agents).

    This FiD agents has a mock retriever that picks the retrieved docs from the observed
    example.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        opt = deepcopy(opt)
        opt['rag_retriever_type'] = RetrieverType.OBSERVATION_ECHO_RETRIEVER.value
        self._n_docs = opt['n_docs']
        if opt['rag_retriever_query'] != 'full_history':
            prev_sel = opt['rag_retriever_query']
            opt['rag_retriever_query'] = 'full_history'
            logging.warning(
                'GoldDocRetrieverFiDAgent only works with `rag_retriever_query` being `"full_history"`. '
                f'Changing opt value for `rag_retriever_query`: `"{prev_sel}"` -> `"full_history"`'
            )
        if not (
            opt['dynamic_batching'] in [None, 'off']
            and opt.get('eval_dynamic_batching') in [None, 'off']
        ):
            raise RuntimeError(
                "For now dynamic batching doesn't work with ObservationEchoRetriever as it cleans up _saved_docs mapping after each batch act."
            )
        super().__init__(opt, shared=shared)

    @abstractmethod
    def get_retrieved_knowledge(self, message):
        """
        Extracts the retrieved knowledge from the message.
        """

    def show_observation_to_echo_retriever(self, observation: Message):
        retrieved_docs = self.get_retrieved_knowledge(observation)
        if len(retrieved_docs) > self._n_docs:
            logging.warning(
                f'Your `get_retrieved_knowledge` method returned {len(retrieved_docs)} Documents, '
                f'instead of the expected {self._n_docs} (set by `--n-docs`). '
                f'This agent will only use the first {self._n_docs} Documents. '
                'Consider modifying your implementation of `get_retrieved_knowledge` to avoid unexpected results. '
                '(or alternatively you may increase `--n-docs` parameter)'
            )
            retrieved_docs = retrieved_docs[: self._n_docs]
        self.model_api.retriever.add_retrieve_doc(
            observation[self._query_key], retrieved_docs
        )

    def _set_query_vec(self, observation: Message) -> Message:
        self.show_observation_to_echo_retriever(observation)
        super()._set_query_vec(observation)

    def batch_act(self, observations):
        """
        Clear the _saved_docs and _query_ids mappings in ObservationEchoRetriever.
        """
        batch_reply = super().batch_act(observations)
        if hasattr(self.model_api.retriever, 'clear_mapping'):
            self.model_api.retriever.clear_mapping()
        return batch_reply


class WizIntGoldDocRetrieverFiDAgent(GoldDocRetrieverFiDAgent):
    """
    Gold knowledge FiD agent for the Wizard of Internet task.
    """

    def _extract_doc_from_message(self, message: Message, idx: int):
        """
        Returns the `idx`-th `__retrieved-docs__` in the `message` as a Document object.
        """
        return Document(
            docid=message[consts.RETRIEVED_DOCS_URLS][idx],
            title=message[consts.RETRIEVED_DOCS_TITLES][idx],
            text=message[consts.RETRIEVED_DOCS][idx],
        )

    def get_retrieved_knowledge(self, message: Message):

        retrieved_docs = []
        if not message.get(consts.RETRIEVED_DOCS):
            return retrieved_docs

        # First adding the docs with selected sentences.
        selected_sentences = message[consts.SELECTED_SENTENCES]
        n_docs_in_message = len(message[consts.RETRIEVED_DOCS])
        already_added_doc_idx = []

        if ' '.join(selected_sentences) == consts.NO_SELECTED_SENTENCES_TOKEN:
            return retrieved_docs  # `retrieved_docs` is empty at this point

        for doc_idx in range(n_docs_in_message):
            doc_content = message[consts.RETRIEVED_DOCS][doc_idx]
            for sel_sentc in selected_sentences:
                if sel_sentc in doc_content and doc_idx not in already_added_doc_idx:
                    retrieved_docs.append(
                        self._extract_doc_from_message(message, doc_idx)
                    )
                    already_added_doc_idx.append(doc_idx)
                    break
            if len(retrieved_docs) == self._n_docs and doc_idx != (self._n_docs - 1):
                logging.warning(
                    f'More than {self._n_docs} documents have selected sentences. Trimming them to the first {self._n_docs}'
                )
                break

        # Then adding other (filler) docs.
        # We add them by iterating forward in the __retrieved-docs__ list for repeatability,
        # but we shuffle the order of the final retruned docs, to make sure model doesn't cheat.
        for doc_idx in range(n_docs_in_message):
            if len(retrieved_docs) == self._n_docs:
                break

            if doc_idx in already_added_doc_idx:
                continue

            retrieved_docs.append(self._extract_doc_from_message(message, doc_idx))

        if n_docs_in_message > len(retrieved_docs):
            logging.debug(
                f'Trimmed retrieved docs from {n_docs_in_message} to {len(retrieved_docs)}'
            )
        random.shuffle(retrieved_docs)
        return retrieved_docs


def concat_enc_outs(
    input: torch.LongTensor,
    enc_out: torch.Tensor,
    mask: torch.BoolTensor,
    embedding_size: int,
    padding_idx: int,
    right_padded: bool = True,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Concatenate Encoder Outputs.

    Does the whole "FiD" thing; each query/document pair is independently encoded in the
    Encoder, so we need to concatenate all the outputs prior to sending to the decoder.

    :param input:
        [bsz, seqlen] original input to the encoder
    :param enc_out:
        [bsz * n_docs, seqlen] output representations from the encoder
    :param mask:
        encoder mask
    :param embedding_size:
        emb/hidden size of the enc representations
    :param padding_idx:
        pad token index; used for mask purposes.
    :param right_padded:
        whether the input is right padded (true) or left padded (false)

    :return (new_out, new_mask):
        return the encoder output and encoder mask, appropriately concatenated.
    """
    bsz, n_docs = input.size(0), enc_out.size(0) // input.size(0)
    split_enc_out = enc_out.split([n_docs] * bsz, dim=0)
    split_mask = mask.split([n_docs] * bsz, dim=0)

    concat_outs: List[torch.Tensor] = []
    concat_lengths = []
    for i in range(bsz):
        mask_i = split_mask[i].view(-1)
        out_i = split_enc_out[i].reshape(-1, embedding_size)[mask_i]
        concat_outs.append(out_i)
        concat_lengths.append(out_i.size(0))

    new_out = enc_out.new(bsz, max(concat_lengths), embedding_size)
    new_mask: torch.BoolTensor = mask.new(bsz, max(concat_lengths))  # type: ignore
    new_out.fill_(padding_idx)
    new_mask.fill_(False)

    for i, (out_i, length_i) in enumerate(zip(concat_outs, concat_lengths)):
        if right_padded:
            new_out[i, :length_i] = out_i
            new_mask[i, :length_i] = True
        else:
            new_out[i, new_out.size(1) - length_i :] = out_i
            new_mask[i, new_out.size(1) - length_i :] = True

    return new_out, new_mask
