#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Module code for BlenderBot2.
"""
import random
import time
import torch
import torch.nn
from typing import List, Tuple, Dict, Optional, Any

from parlai.agents.fid.fid import FidModel, T5FidModel, concat_enc_outs, Fid
from parlai.agents.rag.args import RetrieverType
from parlai.agents.rag.rag import RagModel, T5RagModel
from parlai.agents.rag.dpr import DprQueryEncoder, DprDocumentEncoder
from parlai.agents.rag.retrievers import (
    RagRetriever,
    Document,
    BLANK_DOC,
    argsort_scores_and_docs,
    retriever_factory as rag_retriever_factory,
    RagRetrieverTokenizer,
    SearchQuerySearchEngineRetriever,
    SearchQueryFAISSIndexRetriever,
    ObservationEchoRetriever,
)
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
import parlai.utils.logging as logging
from parlai.utils.torch import padded_tensor

from .sub_modules import (
    QueryGenerator,
    MemoryDecoder,
    clean_vec_with_dict,
    RetrievalType,
    KnowledgeAccessMethod,
)


def retriever_factory(
    opt: Opt, dictionary: DictionaryAgent, shared=None
) -> Optional[RagRetriever]:
    """
    Build retriever.

    Override to build special BB2 Search Retrievers, if necessary

    :param opt:
        ParlAI Opt
    :param dictionary:
        dictionary agent
    :param shared:
        shared objects.

    :return retriever:
        return a retriever for RAG.
    """
    if opt.get('converting'):
        return None
    retriever = RetrieverType(opt['rag_retriever_type'])
    if retriever is RetrieverType.SEARCH_ENGINE:
        return BB2SearchQuerySearchEngineRetriever(opt, dictionary, shared=shared)
    elif retriever is RetrieverType.SEARCH_TERM_FAISS:
        return BB2SearchQueryFaissIndexRetriever(opt, dictionary, shared=shared)
    elif retriever is RetrieverType.OBSERVATION_ECHO_RETRIEVER:
        return BB2ObservationEchoRetriever(opt, dictionary, shared=shared)
    else:
        return rag_retriever_factory(opt, dictionary, shared=shared)


class BlenderBot2RagModel(RagModel):
    """
    BlenderBot 2 RAG Model.

    Employs both a regular retriever and a long-term memory.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, retriever_shared=None):
        from .blenderbot2 import RAG_MODELS

        super().__init__(opt, dictionary, retriever_shared)
        self.opt = opt
        self.dummy_retriever = DummyRetriever(opt, dictionary)
        assert self.retriever is not None
        query_encoder = (
            self.retriever.query_encoder
            if hasattr(self.retriever, 'query_encoder')
            and opt['share_search_and_memory_query_encoder']
            else None
        )
        self.long_term_memory = LongTermMemory(
            opt, dictionary, query_encoder
        )  # type: ignore
        self.query_generator = QueryGenerator(opt)
        self.memory_decoder = MemoryDecoder(opt)

        # attrs
        self._rag_model_interface = RAG_MODELS[self.rag_model_type](opt, self.pad_idx)
        self.knowledge_access_method = KnowledgeAccessMethod(
            opt['knowledge_access_method']
        )
        self.search = RetrieverType(opt['rag_retriever_type']) in [
            RetrieverType.SEARCH_ENGINE,
            RetrieverType.SEARCH_TERM_FAISS,
        ]
        self.should_generate_query = (
            self.knowledge_access_method is KnowledgeAccessMethod.CLASSIFY
            or self.search
        ) and (
            self.knowledge_access_method
            not in [KnowledgeAccessMethod.MEMORY_ONLY, KnowledgeAccessMethod.NONE]
        )

    @classmethod
    def build_retriever(
        cls,
        opt: Opt,
        dictionary: DictionaryAgent,
        retriever_shared: Optional[Dict[str, Any]],
    ) -> Optional[RagRetriever]:
        return retriever_factory(opt, dictionary, retriever_shared)

    def has_query_generator(self) -> bool:
        """
        Return whether there's a query generator.

        Directly access the query generator's agents.
        """
        return bool(self.query_generator.agents)

    def has_memory_decoder(self) -> bool:
        """
        Return whether there is a memory decoder.

        Directly access the memory decoder's agents.
        """
        return bool(self.memory_decoder.agents)

    def tokenize_query_generator_input(self, input: str) -> List[int]:
        """
        Tokenize the input for the query generator.
        """
        assert self.has_query_generator()
        return self.query_generator.tokenize_input(input)

    def tokenize_memory_decoder_input(self, input: str) -> List[int]:
        """
        Tokenize input for memory decoder.
        """
        assert self.has_memory_decoder()
        return self.memory_decoder.tokenize_input(input)

    def tokenize_memory(self, input: str) -> List[int]:
        """
        Tokenize input for Memory Retriever.
        """
        return self.long_term_memory.tokenize_query(input)

    def get_retrieval_type(self) -> torch.LongTensor:
        """
        Return retrieval type for current batch.

        Accesses the query generator directly.
        """
        assert self.has_query_generator()
        return self.query_generator.retrieval_type

    def encoder(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        query_vec: torch.LongTensor,
        input_turns_cnt: torch.LongTensor,
        memory_vec: torch.LongTensor,
        num_memories: torch.LongTensor,
        query_generator_vec: torch.LongTensor,
        gold_doc_vec: torch.LongTensor,
        gold_doc_title_vec: torch.LongTensor,
        num_gold_docs: torch.LongTensor,
        memory_decoder_vec: torch.LongTensor,
        num_memory_decoder_vecs: torch.LongTensor,
        skip_search: torch.BoolTensor,
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
        Override RagModel.encoder to pass along several other input vecs.

        :param input:
            2D [bsz, seqlen] input to the encoder
        :param input_lengths:
            1D [bsz] lengths of each input item
        :param query_vec:
            2D [bsz*n_turns, seqlen] input for the retriever
        :param input_turns_cnt:
            1D [bsz] number of dialogue turns for each input example
        # Begin New Params
        :param memory_vec:
            3D [bsz, num_mems, seqlen] set of memories to write for each batch item
        :param num_memories:
            1D [bsz] # of memories per batch item
        :param query_generator_vec:
            2D [bsz, seqlen] input to the query generator
        :param gold_doc_vec:
            3D [bsz, num_docs, seqlen] gold documents per batch item
        :param gold_doc_title_vec:
            3D [bsz, num_docs, seqlen] gold document titles per batch item
        :param num_gold_docs:
            1D [bsz] # of gold documents per batch item
        :param memory_decoder_vec:
            3D [bsz, num_lines, seqlen] text to convert to memories with memory decoder
        :param num_memory_decoder_vecs:
            1D [bsz] # of memory decoder vectors for each batch item
        :param skip_search:
            1D [bsz] whether to skip search
        """
        # Retrieve, get expanded input
        if all([tensor is not None for tensor in [input_lengths, query_vec]]):
            expanded_input, top_docs, top_doc_scores = self.retrieve_and_concat(
                input,
                input_lengths,
                query_generator_vec,
                query_vec,
                input_turns_cnt,
                memory_vec,
                num_memories,
                gold_doc_vec,
                gold_doc_title_vec,
                num_gold_docs,
                memory_decoder_vec,
                num_memory_decoder_vecs,
                skip_search,
            )
        else:
            expanded_input = input
            top_docs = top_doc_scores = None

        # Run through seq2seq encoder
        tensor, mask = self.seq2seq_encoder(
            expanded_input, positions, segments
        )  # type: ignore

        return tensor, mask, input_turns_cnt, top_docs, top_doc_scores

    def get_retrieval_indices(
        self, ret_vec: torch.LongTensor, ret_type: RetrievalType
    ) -> torch.LongTensor:
        """
        Return the batch indices for the given retrieval type.

        This function is extremely overloaded to handle all `KnowledgeAccessMethod`s and
        all `RetrievalType`s.

        Basically, if BB2's Access Method is not CLASSIFY, we return all indices
        if the specified retrieval type matches the access method. Otherwise, we look
        at the retrieval_type vector to find the corresponding indices.

        :param ret_vec:
            the retrieval_type vector indicating the "classified" retrieval type
            for each batch item
        :param ret_type:
            the retrieval type being considered here.

        :return indices:
            return which batch indices will utilize the given RetrievalType
        """
        no_indices = torch.zeros(0).long()
        all_indices = torch.arange(ret_vec.size(0)).long()
        type_indices = ret_vec.eq(ret_type.value).nonzero().squeeze(1).long()
        assert isinstance(all_indices, torch.LongTensor)
        assert isinstance(no_indices, torch.LongTensor)
        assert isinstance(type_indices, torch.LongTensor)

        if self.knowledge_access_method is KnowledgeAccessMethod.NONE:
            if ret_type is RetrievalType.NONE:
                return all_indices
            else:
                return no_indices
        elif self.knowledge_access_method is KnowledgeAccessMethod.ALL:
            if ret_type is RetrievalType.NONE:
                return no_indices
            else:
                return all_indices
        elif self.knowledge_access_method is KnowledgeAccessMethod.SEARCH_ONLY:
            if ret_type is RetrievalType.SEARCH:
                return all_indices
            else:
                return no_indices
        elif self.knowledge_access_method is KnowledgeAccessMethod.MEMORY_ONLY:
            if ret_type is RetrievalType.MEMORY:
                return all_indices
            else:
                return no_indices
        else:
            assert self.knowledge_access_method is KnowledgeAccessMethod.CLASSIFY
            return type_indices

    def flush_previous_retriever_search_results(self):
        if not hasattr(self, 'retriever'):
            return
        if hasattr(self.retriever, 'top_docs'):
            delattr(self.retriever, 'top_docs')
        if hasattr(self.retriever, 'search_queries'):
            delattr(self.retriever, 'search_queries')

    def retrieve_and_concat(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        query_generator_vec: torch.LongTensor,
        query_vec: torch.LongTensor,
        input_turns_cnt: torch.LongTensor,
        memory_vec: torch.LongTensor,
        num_memories: torch.LongTensor,
        gold_doc_vec: torch.LongTensor,
        gold_doc_title_vec: torch.LongTensor,
        num_gold_docs: torch.LongTensor,
        memory_decoder_vec: torch.LongTensor,
        num_memory_decoder_vecs: torch.LongTensor,
        skip_search: torch.BoolTensor,
    ) -> Tuple[torch.LongTensor, List[List[Document]], torch.Tensor]:
        """
        Override RagModel.retrieve_and_concat to perform different retrieval, depending
        on the RetrieverType.
        """
        self.flush_previous_retriever_search_results()
        start = time.time()
        logging.debug(f'Begin encoder: {time.time() - start:.2f}')
        if input_turns_cnt is not None:
            if query_generator_vec is not None:
                query_generator_vec = query_generator_vec.repeat_interleave(
                    input_turns_cnt, dim=0
                )  # type: ignore
            if memory_vec is not None:
                memory_vec = memory_vec.repeat_interleave(
                    input_turns_cnt, dim=0
                )  # type: ignore
            if num_memories is not None:
                num_memories = num_memories.repeat_interleave(
                    input_turns_cnt, dim=0
                )  # type: ignore
            if memory_decoder_vec is not None:
                memory_decoder_vec = memory_decoder_vec.repeat_interleave(
                    input_turns_cnt, dim=0
                )  # type: ignore
            if num_memory_decoder_vecs is not None:
                num_memory_decoder_vecs = num_memory_decoder_vecs.repeat_interleave(
                    input_turns_cnt, dim=0
                )  # type: ignore
        n_input = (
            input_turns_cnt.sum().item()
            if input_turns_cnt is not None
            else input.size(0)
        )
        # 0a. Classify retrieval type, if necessary
        generated_memories = [[] for _ in range(int(n_input))]
        if memory_decoder_vec is not None:
            generated_memories = self.memory_decoder.generate_memories(
                memory_decoder_vec, num_memory_decoder_vecs
            )
        if self.should_generate_query:
            assert self.has_query_generator()
            retrieval_type, search_queries = self.query_generator.classify_retrieval(
                query_generator_vec, num_memories, generated_memories, skip_search
            )
            logging.debug(f'Classify Retrieval: {time.time() - start:.2f}')
        else:
            retrieval_type = torch.LongTensor(input.size(0))
            search_queries = None

        # 1. Retrieve
        top_docs: List[List[Document]] = [[] for _ in range(int(n_input))]
        doc_scores: List[List[torch.Tensor]] = [[] for _ in range(int(n_input))]

        # 1a. retrieve from faiss or search
        search_indices = self.get_retrieval_indices(
            retrieval_type, RetrievalType.SEARCH
        )
        if search_indices.numel() > 0:
            search_docs, search_doc_scores = self.perform_search(
                search_queries, query_vec, search_indices
            )
            logging.debug(f'Search Complete: {time.time() - start:.2f}')
            logging.debug(f'search: {search_docs}')
            if gold_doc_vec is not None:
                logging.debug(f'num gold docs: {num_gold_docs}')
            self._fill_docs_and_scores(
                top_docs,
                doc_scores,
                search_indices,
                search_docs,
                search_doc_scores,
                gold_doc_vec,
                gold_doc_title_vec,
                num_gold_docs,
            )

        # 1b. memory search
        memory_indices = self.get_retrieval_indices(
            retrieval_type, RetrievalType.MEMORY
        )
        if memory_indices.numel() > 0:
            memories, memory_scores = self.access_long_term_memory(
                query_vec,
                memory_indices,
                memory_vec,
                num_memories,
                memory_decoder_vec,
                generated_memories,
            )
            logging.debug(f'Memory Access Complete: {time.time() - start:.2f}')
            if memories is not None and memory_scores is not None:
                self._fill_docs_and_scores(
                    top_docs, doc_scores, memory_indices, memories, memory_scores
                )

        # 1c. no search
        no_search_indices = self.get_retrieval_indices(
            retrieval_type, RetrievalType.NONE
        )
        if no_search_indices.numel() > 0:
            dummy_docs, dummy_scores = self.dummy_retriever.retrieve(
                query_vec[no_search_indices]  # type: ignore
            )
            logging.debug('no search')
            self._fill_docs_and_scores(
                top_docs, doc_scores, no_search_indices, dummy_docs, dummy_scores
            )

        # 2. Expand the input
        if input_turns_cnt is not None:
            input = input.repeat_interleave(input_turns_cnt, dim=0)  # type: ignore
            input_lengths = input_lengths.repeat_interleave(
                input_turns_cnt, dim=0
            )  # type: ignore

        # Filtering empty doc_scores added due to dynamic batching (if used)
        doc_scores = [[s for s in ds if s is not None] for ds in doc_scores if ds]
        top_doc_scores = torch.stack(
            [torch.cat([s_i for s_i in scores_i]) for scores_i in doc_scores]
        )
        expanded_input = self.concat_docs_and_input(
            input, input_lengths, top_docs, top_doc_scores.size(1)
        )
        return expanded_input, top_docs, top_doc_scores

    def _fill_docs_and_scores(
        self,
        top_docs: List[List[Document]],
        doc_scores: List[List[torch.Tensor]],
        indices: torch.LongTensor,
        docs: List[List[Document]],
        scores: torch.Tensor,
        gold_docs: Optional[torch.LongTensor] = None,
        gold_doc_titles: Optional[torch.LongTensor] = None,
        num_gold: Optional[torch.LongTensor] = None,
    ) -> Tuple[List[List[Document]], List[List[torch.Tensor]]]:
        """
        Fill top docs and doc scores with retrieved documents for L batch indices.

        Documents either come from a retriever or a long-term memory.

        :param top_docs:
            bsz-length list of document sets
        :param doc_scores:
            bsz x n_docs tensor of doc scores
        :param indices:
            batch indices (of length L) to consider
        :param docs:
            L-size list of document sets
        :param scores:
            L-size list of document scores
        :param gold_docs:
            optional list of gold documents to insert into the document set
        :param gold_doc_titles:
            corresponding optional list of doc titles
        :param num_gold:
            how many gold documents are provided per batch item.

        :return top_docs, doc_scores:
            return top_docs and doc_scores with documents and scores filled in
        """
        device = scores.device
        for idx, i in enumerate(indices):
            if num_gold is not None and num_gold[i] > 0:
                assert gold_doc_titles is not None
                assert gold_docs is not None
                replace_inds = random.sample(range(len(docs[idx])), int(num_gold[i]))
                for g_idx, replace in enumerate(replace_inds):
                    # the gold docs are tokenized with the retriever tokenizer.
                    # give the docs the max score, for rag models.
                    docs[idx][replace] = Document(
                        title=self.dict.vec2txt(
                            clean_vec_with_dict(self.dict, gold_doc_titles[i][g_idx])
                        ),
                        text=self.dict.vec2txt(
                            clean_vec_with_dict(self.dict, gold_docs[i][g_idx])
                        ),
                        docid='',
                    )
                    scores[idx][replace] = scores[idx][0].detach()
            try:
                top_docs[i] += docs[idx]
                if self.fp16:
                    scores = scores.half()
                doc_scores[i].append(scores[idx])
            except IndexError:
                # Docs are not provided for this example for this retrieval type.
                # Therefore, we assert that we are in the 'all' mode.
                top_docs[i] += [BLANK_DOC] * self.opt['n_docs']
                doc_scores[i].append(torch.ones(self.opt['n_docs']).to(device))

        return top_docs, doc_scores

    #################################
    # Retrieve/Concat Sub Functions #
    #################################

    def perform_search(
        self,
        search_queries: Optional[List[str]],
        query_vec: torch.LongTensor,
        search_indices: torch.LongTensor,
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Retrieve via retriever from global knowledge.

        :param search_queries:
            if searching, a list of search queries
        :param query_vec:
            query encoder input if retrieving from FAISS
        :param search_indices:
            batch indices to search

        :return docs, scores:
            return the documents and corresponding retrieval scores.
        """
        assert self.retriever is not None
        if self.search:
            assert search_queries
            assert isinstance(
                self.retriever, SearchQuerySearchEngineRetriever
            ) or isinstance(self.retriever, SearchQueryFAISSIndexRetriever)
            self.retriever.set_search_queries(search_queries)
        search_docs, search_doc_scores = self.retriever.retrieve(
            query_vec[search_indices]  # type: ignore
        )
        return search_docs, search_doc_scores

    def access_long_term_memory(
        self,
        query_vec: torch.LongTensor,
        memory_indices: torch.LongTensor,
        memory_vec: Optional[torch.LongTensor],
        num_memories: torch.LongTensor,
        memory_decoder_vec: Optional[torch.LongTensor],
        generated_memories: List[List[str]],
    ) -> Tuple[Optional[List[List[Document]]], Optional[torch.Tensor]]:
        """
        Access long term memory.

        :param query_vec:
            retrieval vector for the long-term memory
        :param memory_indices:
            indices to access memory slots
        :param memory_vec:
            extracted memories from the observation
        :param num_memories:
            bsz-length tensor corresponding to number of memories per batch item
        :param memory_decoder_vec:
            input to the memory decoder
        :param generated_memories:
            memories generated by the memory decoder

        :return memories, memory_scores:
            return memories and memory scores, if there are memories retrieved
        """
        start = time.time()
        memories = None
        memory_scores = None
        memory_dict = {}
        indices = memory_indices.tolist()

        if memory_vec is not None:
            # Only look in memory_vec for batch elements with memories
            memory_ids = [m for m in indices if num_memories[m] > 0]
            memory_dict = {
                batch_id: memory_vec[batch_id, : num_memories[mem_id]]
                for batch_id, mem_id in enumerate(memory_ids)
            }
        if memory_decoder_vec is not None:
            for batch_id in indices:
                new_mems_i = generated_memories[batch_id]
                if not new_mems_i:
                    continue
                tokenized = [
                    self.long_term_memory.tokenize_query(m)
                    for m in generated_memories[batch_id]
                ]
                if batch_id in memory_dict:
                    tokenized += memory_dict[batch_id].tolist()
                new_mems_i, _ = padded_tensor(
                    tokenized, pad_idx=self.dict[self.dict.null_token]  # type: ignore
                )
                memory_dict[batch_id] = new_mems_i.to(query_vec)
        if self.knowledge_access_method in [
            KnowledgeAccessMethod.ALL,
            KnowledgeAccessMethod.MEMORY_ONLY,
        ]:
            # Add dummy memories just in case we are retrieving from memories
            if memory_vec is not None:
                seqlen = memory_vec.size(-1)
            elif memory_decoder_vec is not None:
                seqlen = memory_decoder_vec.size(-1)
            else:
                seqlen = query_vec.size(-1)
            for batch_id in indices:
                if batch_id not in memory_dict:
                    memory_dict[batch_id] = torch.zeros(1, seqlen).to(query_vec)
        if memory_dict:
            # first make sure all memories are padded to the same length.
            max_length = max([m.size(-1) for m in memory_dict.values()])
            for batch_id in memory_dict:
                vec = memory_dict[batch_id]
                if vec.size(-1) < max_length:
                    memory_dict[batch_id] = torch.cat(
                        [
                            vec,
                            torch.zeros((*vec.shape[:-1], max_length - vec.size(-1)))
                            .fill_(self.dict[self.dict.null_token])
                            .to(vec),
                        ],
                        dim=1,
                    )
            self.long_term_memory.write_memory(memory_dict)  # type: ignore
            logging.debug(f'Write Memory Complete: {time.time() - start:.2f}')
        if self.long_term_memory.has_memory():
            memories, memory_scores = self.long_term_memory.retrieve(
                query_vec[memory_indices]  # type: ignore
            )
            logging.debug(f'Memory Retrieval Complete: {time.time() - start:.2f}')
            logging.debug(f'memories: {memories}')
            logging.verbose('Reading from Memory')

        return memories, memory_scores


class T5BlenderBot2RagModel(T5RagModel, BlenderBot2RagModel):
    pass


class DummyRetriever(RagRetriever):
    """
    Dummy Retriever returns blank documents, and equal scores.

    It is utilized to pad the document numbers for mis-matched batch inputs.
    """

    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Simply construct a list of blank documents for each query.
        """

        documents = [[BLANK_DOC] * self.opt['n_docs']] * query.size(0)
        scores = torch.ones(query.size(0), self.opt['n_docs']).to(query.device)
        return documents, scores


class LongTermMemory(RagRetriever):
    """
    The LongTermMEmory writes document embeddings to a memory.

    Retrieval then scores all documents in the memory and returns the final results.
    """

    def __init__(
        self,
        opt: Opt,
        dictionary: DictionaryAgent,
        query_encoder: Optional[torch.nn.Module] = None,
        shared=None,
    ):
        super().__init__(opt, dictionary, shared)
        self.n_docs = opt['n_docs']
        if query_encoder is None:
            self.query_encoder = DprQueryEncoder(
                opt,
                dpr_model=opt['memory_reader_model'],
                pretrained_path=opt['dpr_model_file'],
            )
        else:
            self.query_encoder = query_encoder
        self.memory_encoder = DprDocumentEncoder(
            opt,
            dpr_model=opt['memory_writer_model'],
            pretrained_path=opt['memory_writer_model_file'],
        ).eval()
        self._tokenizer = RagRetrieverTokenizer(
            datapath=opt['datapath'],
            query_model=opt['query_model'],
            dictionary=dictionary,
            delimiter='\n',
            max_length=opt['memory_retriever_truncate']
            if opt['memory_retriever_truncate'] > 0
            else opt['rag_query_truncate'],
        )
        self.max_memories = opt.get('max_memories', 100)
        self.num_memory_slots = opt.get('batchsize', 1) * opt.get('rag_turn_n_turns', 1)
        self.memory_vec_dict: Dict[int, torch.LongTensor] = {  # type: ignore
            k: torch.zeros(self.max_memories, opt['max_doc_token_length']).to(
                torch.int64
            )
            for k in range(self.num_memory_slots)
        }
        self.memory_enc_dict: Dict[int, torch.Tensor] = {
            k: torch.zeros(self.max_memories, opt['retriever_embedding_size'])
            for k in range(self.num_memory_slots)
        }
        self.active_memory_slots: List[int] = []
        self.dict = dictionary

    def has_memory(self) -> bool:
        """
        Return whether there is memory.
        """
        return bool(self.active_memory_slots)

    def write_memory(self, mem_dict: Dict[int, torch.LongTensor]):
        """
        Write vectors to memory.

        Assume that we clear the memory as well.

        :param mem_dict:
            mapping from memory slot to 2D-tokenized memories
        """
        self.active_memory_slots = list(mem_dict.keys())
        with torch.no_grad():
            slot_num_mems = [m.size(0) for m in mem_dict.values()]
            logging.debug(f'Writing {slot_num_mems} memories')
            mem_vecs = torch.cat(list(mem_dict.values()), dim=0)
            mem_encs = self.memory_encoder(mem_vecs)
            offset = 0
            for mem_slot, num_mems in zip(mem_dict.keys(), slot_num_mems):
                self.memory_vec_dict[mem_slot] = mem_vecs[  # type: ignore
                    offset : offset + num_mems
                ]
                self.memory_enc_dict[mem_slot] = mem_encs[offset : offset + num_mems]
                offset += num_mems

    def score_memories(self, query_enc: torch.Tensor) -> List[torch.Tensor]:
        """
        Score memories.
        """
        scores = []
        for i in range(query_enc.size(0)):
            scores.append(
                (
                    query_enc[i : i + 1]
                    @ self.memory_enc_dict[self.active_memory_slots[i]].t()
                ).squeeze(0)
            )
        return scores

    def retrieve_and_score(
        self, query: torch.LongTensor
    ) -> Tuple[List[List[Document]], torch.Tensor]:
        """
        Retrieve and score.

        Encode

        :param query:
            query tokens

        :return (docs, scores):
            docs: list of (text, title) tuples for each batch example
            scores: doc scores
        """
        query_enc = self.query_encoder(query)
        scores = self.score_memories(query_enc)

        top_docs, top_doc_scores = [], []
        for i in range(query.size(0)):
            scores_i = scores[i]
            memories_i, scores_i = argsort_scores_and_docs(
                scores_i, self.memory_vec_dict[i], self.n_docs  # type: ignore
            )
            mem_docs = []
            for mem in memories_i:
                mem_doc = Document('', self._tokenizer.decode(mem), '')  # type: ignore
                mem_doc.TITLE_DELIM = self.opt['memory_doc_title_delimiter']
                mem_docs.append(mem_doc)

            if len(mem_docs) < self.n_docs:
                # add dummy docs
                num_blank = self.n_docs - len(mem_docs)
                mem_docs += [BLANK_DOC] * num_blank
                scores_i = torch.cat([scores_i, torch.zeros(num_blank).to(scores_i)])
            top_docs.append(mem_docs)
            top_doc_scores.append(scores_i)
            logging.debug(scores_i)

        return top_docs, torch.stack(top_doc_scores)


class BlenderBot2Fid(Fid):
    """
    FiD Interface for BB2.
    """

    def __init__(self, opt: Opt, null_idx: int):
        super().__init__(opt, null_idx)
        if (
            KnowledgeAccessMethod(opt['knowledge_access_method'])
            is KnowledgeAccessMethod.ALL
        ):
            # Need to account for memories + search results
            self.n_docs *= 2


class BlenderBot2FidModelMixin:
    embedding_size: int
    pad_idx: int
    long_term_memory: LongTermMemory
    retriever: RagRetriever

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, retriever_shared=None):
        super().__init__(
            opt, dictionary, retriever_shared=retriever_shared
        )  # type: ignore
        self._rag_model_interface = BlenderBot2Fid(
            opt, dictionary[dictionary.null_token]
        )
        self.embedding_size = opt['embedding_size']
        for param in self.long_term_memory.query_encoder.parameters():
            param.requires_grad = False
        for param in self.long_term_memory.memory_encoder.parameters():
            param.requires_grad = False
        for param in self.retriever.parameters():
            param.requires_grad = False

    def encoder(
        self,
        input: torch.LongTensor,
        input_lengths: torch.LongTensor,
        query_vec: torch.LongTensor,
        input_turns_cnt: torch.LongTensor,
        memory_vec: torch.LongTensor,
        num_memories: torch.LongTensor,
        query_generator_vec: torch.LongTensor,
        gold_doc_vec: torch.LongTensor,
        gold_doc_title_vec: torch.LongTensor,
        num_gold_docs: torch.LongTensor,
        memory_decoder_vec: torch.LongTensor,
        num_memory_decoder_vecs: torch.LongTensor,
        skip_search: torch.BoolTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.BoolTensor,
        Optional[torch.LongTensor],
        Optional[List[List[Document]]],
        Optional[torch.Tensor],
    ]:
        enc_out, mask, input_turns_cnt, top_docs, top_doc_scores = super().encoder(  # type: ignore
            input,
            input_lengths,
            query_vec,
            input_turns_cnt,
            memory_vec,
            num_memories,
            query_generator_vec,
            gold_doc_vec,
            gold_doc_title_vec,
            num_gold_docs,
            memory_decoder_vec,
            num_memory_decoder_vecs,
            skip_search,
            positions,
            segments,
        )  # type: ignore

        if input_turns_cnt is not None:
            # Input Turns is a tensor of dim [bsz]
            input = input.repeat_interleave(input_turns_cnt, dim=0)  # type: ignore

        new_out, new_mask = concat_enc_outs(
            input, enc_out, mask, self.embedding_size, self.pad_idx
        )

        return new_out, new_mask, input_turns_cnt, top_docs, top_doc_scores


class BlenderBot2FidModel(BlenderBot2FidModelMixin, BlenderBot2RagModel, FidModel):
    pass


class T5BlenderBot2FidModel(
    BlenderBot2FidModelMixin, T5FidModel, T5BlenderBot2RagModel
):
    pass


class BB2SearchRetrieverMixin:
    """
    Mixin for BB2 Search Modules.
    """

    def set_search_queries(self, queries: List[str]):
        self.search_queries = queries

    def init_search_query_generator(self, opt):
        pass

    def generate_search_query(self, query: torch.LongTensor) -> List[str]:
        return self.search_queries


class BB2SearchQuerySearchEngineRetriever(
    BB2SearchRetrieverMixin, SearchQuerySearchEngineRetriever
):
    """
    Override Search Engine Retriever to accommodate SQ Generator from BB2 Setup.
    """


class BB2SearchQueryFaissIndexRetriever(
    BB2SearchRetrieverMixin, SearchQueryFAISSIndexRetriever
):
    """
    Override Search Engine Retriever to accommodate SQ Generator from BB2 Setup.
    """


class BB2ObservationEchoRetriever(BB2SearchRetrieverMixin, ObservationEchoRetriever):
    """
    A retriever that reads retrieved docs as part of the observed example message.

    Provides backwards compatibility with BB2 models by instantiating a query encoder.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, shared=None):
        super().__init__(opt, dictionary, shared)
        self.query_encoder = DprQueryEncoder(
            opt, dpr_model=opt['query_model'], pretrained_path=opt['dpr_model_file']
        )
