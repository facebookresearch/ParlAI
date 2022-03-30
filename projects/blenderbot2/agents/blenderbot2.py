#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BlenderBot2 Agent Code.

BlenderBot 2 combines a long-term memory module with a retriever module.

The Search Query Generator generates a query that tells BB2 to either access
its memory or access the internet.

The Memory Decoder examines the context and generates memories to write to
the long-term memory module.
"""
from abc import abstractmethod
import re
import copy
import torch
import torch.nn
import torch.nn.functional as F
from typing import Union, Dict, List, Tuple, Optional, Any

from parlai.agents.fid.fid import FidAgent, WizIntGoldDocRetrieverFiDAgent
from parlai.agents.rag.args import DPR_ZOO_MODEL, QUERY_MODEL_TYPES, RetrieverType
from parlai.agents.rag.rag import RagAgent
from parlai.agents.rag.model_types import (
    RagTurn,
    RagSequence,
    RagToken,
    RagModelInterface,
)
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import Batch, History
from parlai.tasks.wizard_of_internet.constants import (
    SELECTED_DOCS,
    SELECTED_DOCS_TITLES,
    SELECTED_SENTENCES,
    NO_SELECTED_DOCS_TOKEN,
    SKIP_SEARCH,
)
from parlai.utils.torch import padded_3d
from parlai.utils.typing import TShared

from .modules import (
    BlenderBot2RagModel,
    T5BlenderBot2RagModel,
    BlenderBot2FidModel,
    T5BlenderBot2FidModel,
)
from .sub_modules import RetrievalType, KnowledgeAccessMethod
from parlai.agents.fid.fid import SearchQuerySearchEngineFiDAgent


ZOO_QUERY_GENERATOR = 'zoo:blenderbot2/query_generator/model'
ZOO_MEMORY_DECODER = 'zoo:blenderbot2/memory_decoder/model'


class HistoryCleanReply(History):
    def __init__(
        self,
        opt,
        field='text',
        maxlen=None,
        size=-1,
        p1_token='__p1__',
        p2_token='__p2__',
        dict_agent=None,
    ):
        super().__init__(
            opt,
            field=field,
            maxlen=maxlen,
            size=size,
            p1_token=p1_token,
            p2_token=p2_token,
            dict_agent=dict_agent,
        )
        self.add_cleaned_reply_to_history = opt.get(
            'add_cleaned_reply_to_history', False
        )

    @abstractmethod
    def _clean_text(self, txt):
        """
        Clean text to be override with custom logic.
        """

    def add_reply(self, text):
        clean_text = text
        if self.add_cleaned_reply_to_history:
            clean_text = self._clean_text(text)
        super().add_reply(clean_text)


class HistoryCleanUnsafeToken(HistoryCleanReply):
    """
    Override the history _clean_text to filter out special tokens like
    _potentially_unsafe.
    """

    def _clean_text(self, txt):
        cleaned_txt = re.sub(r'_[\S]*unsafe_*', '', txt, flags=re.IGNORECASE)
        return cleaned_txt.strip()


class BlenderBot2ModelTypeMixin(RagModelInterface):
    """
    Override Normal RAG Model Types, in case we retrieve from both memory and search.
    """

    def __init__(self, opt: Opt, null_idx: int):
        super().__init__(opt, null_idx)
        if (
            KnowledgeAccessMethod(opt['knowledge_access_method'])
            is KnowledgeAccessMethod.ALL
        ):
            self.n_docs *= 2


class BlenderBot2RagSequence(BlenderBot2ModelTypeMixin, RagSequence):
    def augment_batch_for_generation(
        self, batch: Batch, model: BlenderBot2RagModel
    ) -> Batch:
        """
        Augment batch for generation.

        For RAG Sequence, we retrieve prior to generation, as we do not consider the
        document probabilities until after generating all of the beams.

        :param batch:
            batch to augment
        :param model:
            model to possibly help with augmenting

        :return batch:
            return batch with text vec swapped out.
        """
        (expanded_input, _, doc_scores) = model.retrieve_and_concat(
            batch.text_vec,
            batch.text_vec.ne(self.null_idx).sum(1),
            batch.query_generator_vec,
            batch.query_vec,
            batch.input_turn_cnt_vec,
            batch.memory_vec,
            batch.num_memories,
            batch.gold_doc_vec,
            batch.gold_doc_title_vec,
            batch.num_gold_docs,
            batch.memory_decoder_vec,
            batch.num_memory_decoder_vecs,
            batch.skip_search,
        )
        doc_log_probs = F.log_softmax(doc_scores, dim=1)
        batch.src_text_vec = batch.text_vec
        batch.text_vec = expanded_input
        batch.doc_log_probs = doc_log_probs
        batch.batchsize = batch.text_vec.size(0)

        return batch

    def get_generation_input(
        self, batch: Batch
    ) -> Tuple[
        torch.LongTensor,
        Optional[torch.LongTensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        For RAG Sequence, we retrieve prior to generation.
        """
        assert batch.text_vec is not None
        return (
            batch.text_vec,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class BlenderBot2RagToken(BlenderBot2ModelTypeMixin, RagToken):
    pass


class BlenderBot2RagTurn(BlenderBot2ModelTypeMixin, RagTurn):
    pass


RAG_MODELS = {
    'sequence': BlenderBot2RagSequence,
    'token': BlenderBot2RagToken,
    'turn': BlenderBot2RagTurn,
}


class BlenderBot2RagAgent(RagAgent):
    """
    Subclass RagAgent to provide BlenderBot2Model with appropriate inputs (specifically,
    memory vectors).
    """

    model: BlenderBot2RagModel

    ##########################
    # Housekeeping functions #
    ##########################
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add RAG Args.
        """
        RagAgent.add_cmdline_args(parser, partial_opt)
        SearchQuerySearchEngineFiDAgent.add_cmdline_args(parser, partial_opt)
        bb2_group = parser.add_argument_group('BlenderBot2 Args')
        bb2_group.add_argument(
            '--knowledge-access-method',
            type=str,
            default=KnowledgeAccessMethod.CLASSIFY.value,
            choices=[r.value for r in KnowledgeAccessMethod],
            help='How to access knowledge for BlenderBot2 '
            'classify => classify the input text, determine which knowledge to access\n'
            'memory_only => only access memories\n'
            'search_only => only access search\n'
            'all => for each input, access from memories and search\n'
            'none => do not access any knowledge.\n',
        )
        bb2_group.add_argument(
            '--memory-key',
            type=str,
            default='full_text',
            help='Field in the observation from which to read memories.',
        )
        bb2_group.add_argument(
            '--query-generator-key',
            type=str,
            default='full_text',
            help='Field for input to the knowledge access classifier.',
        )
        bb2_group.add_argument(
            '--gold-document-key',
            type=str,
            default=SELECTED_DOCS,
            help='Field for selected docs.',
        )
        bb2_group.add_argument(
            '--gold-sentence-key',
            type=str,
            default=SELECTED_SENTENCES,
            help='Field for selected sentences',
        )
        bb2_group.add_argument(
            '--gold-document-titles-key',
            type=str,
            default=SELECTED_DOCS_TITLES,
            help='Field for selected docs titles.',
        )
        bb2_group.add_argument(
            '--skip-search-key',
            type=str,
            default=SKIP_SEARCH,
            help='Field for whether to skip search or not.',
        )
        bb2_group.add_argument(
            '--insert-gold-docs',
            type='bool',
            default=False,
            help='Set true to insert gold docs into retrieved docs.',
        )
        bb2_group.add_argument(
            '--memory-extractor-phrase',
            type=str,
            default='persona:',
            help="phrase used to extract memories from `--memory-key` in the observation. "
            "For example, set to 'your persona:' to limit memories to only lines that "
            "contain 'your persona:'",
        )
        bb2_group.add_argument(
            '--retriever-ignore-phrase',
            type=str,
            default='persona:',
            help='filter input to the global knowledge retriever such that any utterance containing '
            'the phrase will not be given as input.',
        )
        q_gen_group = parser.add_argument_group('BlenderBot2 Query Generator Args')
        q_gen_group.add_argument(
            '--query-generator-ignore-phrase',
            type=str,
            default='persona:',
            help='filter input to the query generator such that any utterance containing '
            'the phrase will not be given as input.',
        )
        q_gen_group.add_argument(
            '--query-generator-model-file',
            type=str,
            default=ZOO_QUERY_GENERATOR,
            help='path to a query generator; specify if searching OR classifying inputs.',
        )
        q_gen_group.add_argument(
            '--query-generator-delimiter',
            type=str,
            default='\n',
            help='delimiter for the query generator',
        )
        q_gen_group.add_argument(
            '--query-generator-inference',
            type=str,
            default='beam',
            help='query generator inference type',
        )
        q_gen_group.add_argument(
            '--query-generator-beam-size', type=int, default=1, help='SQ Gen Beam Size'
        )
        q_gen_group.add_argument(
            '--query-generator-beam-min-length',
            type=int,
            default=2,
            help='SQ Gen Beam Min Length',
        )
        q_gen_group.add_argument(
            '--query-generator-truncate',
            type=int,
            default=-1,
            help='Specify >0 for truncation to SQ generator',
        )
        bb2_group.add_argument(
            '--memory-retriever-truncate',
            type=int,
            default=-1,
            help='Specify >0 for truncation to the memory retriever.',
        )
        bb2_group.add_argument(
            '--retriever-delimiter',
            type=str,
            default='\n',
            help='delimiter for the retriever',
        )
        bb2_group.add_argument(
            '--share-search-and-memory-query-encoder',
            type='bool',
            default=False,
            help='if true, query encoder is shared between search and memory retrievers.',
        )
        bb2_group.add_argument(
            '--memory-reader-model',
            type=str,
            default=None,
            choices=QUERY_MODEL_TYPES,
            help='Model for accessing the memory',
        )
        bb2_group.add_argument(
            '--memory-doc-title-delimiter',
            type=str,
            default=' / ',
            help='title delimiter for memory docs',
        )
        bb2_group.add_argument(
            '--memory-writer-model',
            type=str,
            default='bert',
            hidden=True,
            help='model for writing the memories',
        )
        bb2_group.add_argument(
            '--memory-writer-model-file',
            type=str,
            default=DPR_ZOO_MODEL,
            hidden=True,
            help='model file for memory writer',
        )
        bb2_group.add_argument(
            '--add-cleaned-reply-to-history',
            type=bool,
            default=False,
            help='whether to add the cleaned bb2 generated text without any special tokens to its history',
        )
        memory_decoder = parser.add_argument_group('BlenderBot2 Memory Decoder Args')
        memory_decoder.add_argument(
            '--memory-decoder-key',
            type=str,
            default='full_text',
            help='key of the observation for the memory decoder',
        )
        memory_decoder.add_argument(
            '--memory-decoder-ignore-phrase',
            type=str,
            default='persona:',
            help='filter input to the memory decoder such that any utterance containing '
            'the phrase will not be given as input.',
        )
        memory_decoder.add_argument(
            '--memory-decoder-model-file',
            type=str,
            default=ZOO_MEMORY_DECODER,
            help='path to a memory decoder.',
        )
        memory_decoder.add_argument(
            '--memory-decoder-delimiter',
            type=str,
            default='\n',
            help='delimiter for the memory decoder',
        )
        memory_decoder.add_argument(
            '--memory-decoder-beam-size',
            type=int,
            default=3,
            help='memory decoder Beam Size',
        )
        memory_decoder.add_argument(
            '--memory-decoder-beam-min-length',
            type=int,
            default=10,
            help='memory decoder Beam Min Length',
        )
        memory_decoder.add_argument(
            '--memory-decoder-truncate',
            type=int,
            default=-1,
            help='Specify >0 for truncation to memory decoder',
        )
        memory_decoder.add_argument(
            '--memory-decoder-one-line-memories',
            type='bool',
            default=False,
            help='specify to combine memories on one line, rather than several.',
        )
        return parser

    @classmethod
    def history_class(cls):
        return HistoryCleanUnsafeToken

    @property
    def rag_model_type(self) -> str:
        return self._rag_model_type

    @rag_model_type.setter
    def rag_model_type(self, model: str):
        self._rag_model_type = model
        self._rag_model_interface = RAG_MODELS[model](self.opt, self.NULL_IDX)

    def build_model(self) -> BlenderBot2RagModel:
        """
        Build and return BlenderBot2RagModel.
        """
        if self.generation_model == 't5':
            model = T5BlenderBot2RagModel(self.opt, self.dict)
        else:
            model = BlenderBot2RagModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    @classmethod
    def upgrade_opt(cls, opt_from_disk: Opt):
        # call the parent upgrades
        opt_from_disk = super().upgrade_opt(opt_from_disk)

        if 'memory_doc_delimiter' not in opt_from_disk:
            # 2020-06-22 old delimiter was ':'
            opt_from_disk['memory_doc_delimiter'] = ':'

        return opt_from_disk

    @staticmethod
    def update_state_dict(
        opt: Opt, state_dict: Dict[str, torch.Tensor], model: torch.nn.Module
    ):
        """
        Override RagAgent.update_state_dict to store long term memory state.
        """
        state_dict = RagAgent.update_state_dict(opt, state_dict, model)
        # 1. Retriever state
        if not [k for k in state_dict if 'long_term_memory' in k]:
            long_term_memory_state = {
                f"long_term_memory.{k}": v
                for k, v in model.long_term_memory.state_dict().items()  # type: ignore
            }
            state_dict.update(long_term_memory_state)
        return state_dict

    ###############################
    # Text/Tokenization Overrides #
    ###############################
    def observe(self, observation: Union[Dict, Message]) -> Message:
        """
        Overrides TA.observe to tokenize various additional vectors.
        """
        observation = super().observe(observation)
        if 'memory_vec' not in observation and self.opt['memory_key'] in observation:
            self._set_memory_vec(observation)
        if (
            'query_generator_vec' not in observation
            and self.opt['query_generator_key'] in observation
        ):
            self._set_query_generator_vec(observation)
        if 'gold_doc_vec' not in observation and all(
            k in observation
            for k in [
                self.opt['gold_document_key'],
                self.opt['gold_sentence_key'],
                self.opt['gold_document_titles_key'],
            ]
        ):
            self._set_gold_doc_vec(observation)
        if (
            'memory_decoder_vec' not in observation
            and self.opt['memory_decoder_key'] in observation
        ):
            self._set_memory_decoder_vec(observation)
        return observation

    def _filter_text(self, text: str, filter_phrase: str, delimiter: str = '\n') -> str:
        """
        Filter text such that utterances containing a filter phrase are removed.

        :param text:
            text to filter
        :param filter_phrase:
            phrase on which to filter
        :param delimiter:
            optional extra delimiter on which to split

        :return text:
            return the text after filtering (including or excluding) turns with the filter phrase.
        """
        split_text = [
            t
            for tt in text.split(self.opt.get('delimiter', '\n'))
            for t in tt.split('\n')
        ]
        turns = [t for t in split_text if filter_phrase not in t]
        if not turns:
            new_text = text
        else:
            new_text = delimiter.join(turns)
        return new_text

    def _remove_person_tokens(self, text: str) -> str:
        """
        Remove person tokens from a text input.
        """
        return text.replace(f'{self.P1_TOKEN} ', '').replace(f'{self.P2_TOKEN} ', '')

    def _set_query_vec(self, observation: Message) -> Message:
        """
        Override RAG.set_query_vec to optionally filter phrases.
        """
        query_str = observation[self._query_key]
        if self.opt['retriever_ignore_phrase']:
            query_str = self._filter_text(
                query_str,
                self.opt['retriever_ignore_phrase'],
                delimiter=self.opt['retriever_delimiter'],
            )
        if self.add_person_tokens:
            query_str = self._remove_person_tokens(query_str)
        observation['query_vec'] = self.model_api.tokenize_query(query_str)
        return observation

    def _set_memory_vec(self, observation: Message) -> Message:
        """
        Tokenize the memories for use in long-term memory scoring.

        :param observation:
            observation with input text.

        :return observation:
            return observation with memory vec.
        """
        mem_vecs = None
        method = KnowledgeAccessMethod(self.opt['knowledge_access_method'])
        if method in [
            KnowledgeAccessMethod.ALL,
            KnowledgeAccessMethod.CLASSIFY,
            KnowledgeAccessMethod.MEMORY_ONLY,
        ]:
            memories = observation[self.opt['memory_key']]
            if isinstance(memories, str):
                memories = [
                    t
                    for tt in memories.split(self.opt.get('delimiter', '\n'))
                    for t in tt.split('\n')
                ]
            assert isinstance(memories, list)
            if self.opt['memory_extractor_phrase']:
                # extract text lines only containing the memory extractor phrase
                memories = [
                    m for m in memories if self.opt['memory_extractor_phrase'] in m
                ]
            if memories:
                mem_vecs = [self.model_api.tokenize_memory(mem) for mem in memories]

        observation['memory_vec'] = mem_vecs
        return observation

    def _set_query_generator_vec(self, observation: Message) -> Message:
        """
        Tokenize text for use in the query generator.

        :param observation:
            observation with input text.

        :return observation:
            return observation with query generator vec.
        """
        query_generator_vec = None
        method = KnowledgeAccessMethod(self.opt['knowledge_access_method'])
        if (
            method
            in [
                KnowledgeAccessMethod.ALL,
                KnowledgeAccessMethod.CLASSIFY,
                KnowledgeAccessMethod.SEARCH_ONLY,
            ]
            and self.model_api.has_query_generator()
        ):
            query_generator_input = observation[self.opt['query_generator_key']]
            if self.opt['query_generator_ignore_phrase']:
                query_generator_input = self._filter_text(
                    query_generator_input,
                    self.opt['query_generator_ignore_phrase'],
                    self.opt['query_generator_delimiter'],
                )
            if self.add_person_tokens:
                query_generator_input = self._remove_person_tokens(
                    query_generator_input
                )
            query_generator_vec = self.model_api.tokenize_query_generator_input(
                query_generator_input
            )

        observation['query_generator_vec'] = query_generator_vec
        return observation

    def _set_gold_doc_vec(self, observation: Message) -> Message:
        """
        Tokenize the gold documents, in case we want to include in retrieved documents.

        We chunk up the docs and try to find the chunk that contains the selected sentence.

        If we can't find it, we just use the first chunk.

        :param observation:
            observation with input text.

        :return observation:
            return observation with gold doc vec.
        """
        gold_docs = observation[self.opt['gold_document_key']]
        if not gold_docs or gold_docs == [NO_SELECTED_DOCS_TOKEN]:
            return observation
        doc_vecs = None
        doc_title_vecs = None
        method = KnowledgeAccessMethod(self.opt['knowledge_access_method'])
        chunk_len = self.opt.get("splitted_chunk_length", 256)
        if method in [
            KnowledgeAccessMethod.ALL,
            KnowledgeAccessMethod.CLASSIFY,
            KnowledgeAccessMethod.SEARCH_ONLY,
        ]:
            selected_documents = observation[self.opt['gold_document_key']]
            sentences = observation[self.opt['gold_sentence_key']]
            document_titles = observation[self.opt['gold_document_titles_key']]
            if isinstance(selected_documents, str):
                selected_documents = [selected_documents]
            assert isinstance(selected_documents, list)

            documents = []
            for doc in selected_documents:
                # Try to find the chunk with the selected sentence
                used_chunk = None
                words = doc.split(' ')
                chunks = [
                    ' '.join(words[i : i + chunk_len])
                    for i in range(0, len(words), chunk_len)
                ]
                for chunk in chunks:
                    if any(s in chunk for s in sentences):
                        used_chunk = chunk
                        break
                if not used_chunk:
                    used_chunk = chunks[0]
                documents.append(used_chunk)

            if documents:
                doc_vecs = [self.dict.txt2vec(doc) for doc in documents]
                doc_title_vecs = [self.dict.txt2vec(title) for title in document_titles]

        observation['gold_doc_vec'] = doc_vecs
        observation['gold_doc_title_vec'] = doc_title_vecs
        return observation

    def _set_memory_decoder_vec(self, observation: Message) -> Message:
        """
        Tokenize the input to the memory decoder.

        :param observation:
            observation with input text.

        :return observation:
            return observation with memory vec.
        """
        memory_decoder_vec = None
        method = KnowledgeAccessMethod(self.opt['knowledge_access_method'])
        if (
            method
            in [
                KnowledgeAccessMethod.ALL,
                KnowledgeAccessMethod.CLASSIFY,
                KnowledgeAccessMethod.MEMORY_ONLY,
            ]
            and self.model_api.has_memory_decoder()
        ):
            memory_decoder_input = observation[self.opt['memory_decoder_key']]
            if self.opt['memory_decoder_ignore_phrase']:
                memory_decoder_input = self._filter_text(
                    memory_decoder_input,
                    self.opt['memory_decoder_ignore_phrase'],
                    self.opt['memory_decoder_delimiter'],
                )
            if self.add_person_tokens:
                memory_decoder_input = self._remove_person_tokens(memory_decoder_input)
            conv_lines = [
                t
                for tt in memory_decoder_input.split(self.opt.get('delimiter', '\n'))
                for t in tt.split('\n')
            ]
            memory_decoder_vec = [
                self.model_api.tokenize_memory_decoder_input(i) for i in conv_lines
            ]

        observation['memory_decoder_vec'] = memory_decoder_vec
        return observation

    def batchify(self, obs_batch: List[Message], sort: bool = False) -> Batch:
        """
        Overrides RagAgent.batchify to add several input vectors.
        """
        batch = super().batchify(obs_batch, sort)
        valid_exs = [ex for ex in obs_batch if self.is_valid(ex)]
        batch.memory_vec = None
        batch.num_memories = None
        batch.query_generator_vec = None
        batch.gold_doc_vec = None
        batch.gold_doc_title_vec = None
        batch.num_gold_docs = None
        batch.memory_decoder_vec = None
        batch.num_memory_decoder_vecs = None
        batch.skip_search = None
        if any(ex.get('memory_vec') is not None for ex in valid_exs):
            batch = self._set_batch_memory_vec(valid_exs, batch)
        if any(ex.get('query_generator_vec') is not None for ex in valid_exs):
            batch = self._set_batch_query_generator_vec(valid_exs, batch)
        if any(ex.get('gold_doc_vec') is not None for ex in valid_exs):
            batch = self._set_batch_gold_doc_vec(valid_exs, batch)
        if any(ex.get('memory_decoder_vec') is not None for ex in valid_exs):
            batch = self._set_batch_memory_decoder_vec(valid_exs, batch)
        if any(ex.get(self.opt.get('skip_search_key')) is not None for ex in valid_exs):
            batch = self._set_batch_skip_search(valid_exs, batch)
        return batch

    def _set_batch_memory_vec(self, valid_exs: List[Message], batch: Batch) -> Batch:
        """
        Set the memory vec for the batch.
        """
        mems = []
        num_mems = []
        for ex in valid_exs:
            if ex.get('memory_vec') is not None:
                ms, _ = self._pad_tensor(ex['memory_vec'])
                mems.append(ms)
                num_mems.append(len(ex['memory_vec']))
            else:
                num_mems.append(0)
        batch.memory_vec = padded_3d(mems)
        batch.num_memories = torch.LongTensor(num_mems)
        return batch

    def _set_batch_query_generator_vec(
        self, valid_exs: List[Message], batch: Batch
    ) -> Batch:
        """
        Set the query generator vec for the batch.
        """
        _q_gens = [ex.get('query_generator_vec', self.EMPTY) for ex in valid_exs]
        q_gen_vecs, _lens = self._pad_tensor(_q_gens)
        batch.query_generator_vec = q_gen_vecs
        return batch

    def _set_batch_gold_doc_vec(self, valid_exs: List[Message], batch: Batch) -> Batch:
        """
        Set the gold docs vecs for the batch.
        """
        docs = []
        titles = []
        num_docs = []
        for ex in valid_exs:
            if ex.get('gold_doc_vec') is not None:
                ds, _ = self._pad_tensor(ex['gold_doc_vec'])
                ts, _ = self._pad_tensor(ex['gold_doc_title_vec'])
                docs.append(ds)
                titles.append(ts)
                num_docs.append(len(ex['gold_doc_vec']))
            else:
                docs.append(self.EMPTY.unsqueeze(0))
                titles.append(self.EMPTY.unsqueeze(0))
                num_docs.append(0)
        batch.gold_doc_vec = padded_3d(docs)
        batch.gold_doc_title_vec = padded_3d(titles)
        batch.num_gold_docs = torch.LongTensor(num_docs)
        return batch

    def _set_batch_memory_decoder_vec(
        self, valid_exs: List[Message], batch: Batch
    ) -> Batch:
        """
        Set the memory decoder vec for the batch.
        """
        memory_dec_toks = []
        num_memory_dec_toks = []
        for ex in valid_exs:
            if ex.get('memory_decoder_vec') is not None:
                p_sum_vecs, _lens = self._pad_tensor(ex['memory_decoder_vec'])
                memory_dec_toks.append(p_sum_vecs)
                num_memory_dec_toks.append(len(ex['memory_decoder_vec']))
            else:
                num_memory_dec_toks.append(0)
        batch.memory_decoder_vec = padded_3d(memory_dec_toks)
        batch.num_memory_decoder_vecs = torch.LongTensor(num_memory_dec_toks)
        return batch

    def _set_batch_skip_search(self, valid_exs: List[Message], batch: Batch) -> Batch:
        skip_search = [ex.get(self.opt['skip_search_key'], False) for ex in valid_exs]
        batch.skip_search = torch.BoolTensor(skip_search)
        return batch

    def eval_step(self, batch):
        output = super().eval_step(batch)
        if output is None or not hasattr(self.model_api, 'retriever'):
            return output
        if hasattr(self.model_api.retriever, 'top_docs'):
            output.top_docs = self.model_api.retriever.top_docs
        if hasattr(self.model_api.retriever, 'search_queries'):
            output.search_queries = self.model_api.retriever.search_queries
        if hasattr(self.model_api.memory_decoder, 'memories_full_list'):
            output.memories = self.model_api.memory_decoder.memories_full_list
        return output

    def _model_input(
        self, batch: Batch
    ) -> Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.BoolTensor,
    ]:
        """
        Override RagAgent._model_input to include several more input vectors.

        See BlenderBot2RagModel.encoder for details.
        """
        return (
            batch.text_vec,
            batch.text_vec.ne(self.NULL_IDX).sum(1),
            batch.query_vec,
            batch.input_turn_cnt_vec,
            batch.memory_vec,
            batch.num_memories,
            batch.query_generator_vec,
            batch.gold_doc_vec,
            batch.gold_doc_title_vec,
            batch.num_gold_docs,
            batch.memory_decoder_vec,
            batch.num_memory_decoder_vecs,
            batch.skip_search,
        )

    def compute_loss(
        self, batch: Batch, return_output: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Override Rag.compute_loss to add some additional metrics.
        """
        loss, output = super().compute_loss(batch, return_output=True)
        assert isinstance(self.model_api, BlenderBot2RagModel)
        if (
            KnowledgeAccessMethod(self.opt['knowledge_access_method'])
            is KnowledgeAccessMethod.CLASSIFY
            and self.model_api.has_query_generator()
        ):
            _scores, _preds, enc_state, *_ = output
            _, _, input_turns_cnt, _, _ = enc_state
            retrieval_type = self.model_api.get_retrieval_type()
            assert isinstance(retrieval_type, torch.Tensor)
            if input_turns_cnt is not None:
                new_ret_type = torch.zeros(input_turns_cnt.size(0))
                offset = 0
                for i in range(input_turns_cnt.size(0)):
                    new_ret_type[i] = retrieval_type[offset]
                    offset += input_turns_cnt[i]
                retrieval_type = new_ret_type
            self.record_local_metric(
                'search_class',
                AverageMetric.many(
                    retrieval_type.eq(RetrievalType.SEARCH.value).int().tolist(),
                    [1] * retrieval_type.size(0),
                ),
            )
            self.record_local_metric(
                'memory_class',
                AverageMetric.many(
                    retrieval_type.eq(RetrievalType.MEMORY.value).int().tolist(),
                    [1] * retrieval_type.size(0),
                ),
            )
            self.record_local_metric(
                'none_class',
                AverageMetric.many(
                    retrieval_type.eq(RetrievalType.NONE.value).int().tolist(),
                    [1] * retrieval_type.size(0),
                ),
            )
        if return_output:
            return loss, output
        else:
            return loss


class BlenderBot2FidAgent(FidAgent, BlenderBot2RagAgent):
    model: BlenderBot2FidModel

    def build_model(self) -> Union[BlenderBot2FidModel, T5BlenderBot2FidModel]:
        if self.generation_model == 't5':
            model = T5BlenderBot2FidModel(self.opt, self.dict)
        else:
            model = BlenderBot2FidModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model


class BlenderBot2SearchQueryFiDAgent(BlenderBot2FidAgent):
    def __init__(self, opt: Opt, shared: TShared = None):
        opt = copy.deepcopy(opt)
        opt['rag_retriever_type'] = RetrieverType.SEARCH_ENGINE.value
        super().__init__(opt, shared=shared)


class BlenderBot2WizIntGoldDocRetrieverFiDAgent(
    WizIntGoldDocRetrieverFiDAgent, BlenderBot2FidAgent
):
    pass
