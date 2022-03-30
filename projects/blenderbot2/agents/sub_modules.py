#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
SubModule code for BlenderBot2.

Contains implementations of the Query Generator and Memory Decoder.
"""
from enum import Enum, auto
import os
import string
import time
import torch
import torch.nn
from typing import List, Tuple, Dict, Optional, Any

from parlai.agents.rag.retrievers import clean_vec
from parlai.core.agents import create_agent_from_model_file, create_agent_from_shared
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.torch_agent import TorchAgent
from parlai.tasks.msc.agents import NOPERSONA
import parlai.utils.logging as logging


# The below are the classification outputs mapping to either
# retrieving from memory, or not retrieving at all.
MEMORY_STRINGS = ['convai2', 'personal_knowledge']
NONE_STRINGS = [
    'blended_skill_talk',
    'empathetic_dialogues',
    'dummy',
    'no_passages_used',
]


def strip_punc(s):
    return s.translate(str.maketrans('', '', string.punctuation.replace('_', '')))


def clean_vec_with_dict(dict: DictionaryAgent, vec: torch.LongTensor) -> List[int]:
    """
    Clean the specified vector with the specified dictionary.

    See `parlai.agents.rag.retrievers.clean_vec`for a description
    """
    return clean_vec(
        vec,
        dict[dict.end_token],  # type: ignore
        special_toks=[
            dict[dict.null_token],  # type: ignore
            dict[dict.start_token],  # type: ignore
            dict[dict.end_token],  # type: ignore
            dict[dict.unk_token],  # type: ignore
        ],
    )


class RetrievalType(Enum):
    """
    Retrieval Type indicates the "type" of retrieval.

    That is, we either don't retrieve; retrieve from memory; or retrieve via search.
    """

    NONE = auto()
    SEARCH = auto()
    MEMORY = auto()


class KnowledgeAccessMethod(Enum):
    """
    How BlenderBot2 should retrieve for each input.

    classify => classify the input text, determine which retrieval to use

    memory_only => only retrieve via memories (i.e., from dialogue context)
    search_only => only retrieve via internet/FAISS search
    all => for each input, retrieve both from memories and internet/FAISS search
    none => do not retrieve anything.
    """

    CLASSIFY = 'classify'
    MEMORY_ONLY = 'memory_only'
    SEARCH_ONLY = 'search_only'
    ALL = 'all'
    NONE = 'none'


class BB2SubmoduleMixin:
    """
    Mixin for agents used within BB2.

    agents: list of agents
    agent_dict: dictionary for the agent
    input_type: for logging purposes.
    """

    agents: List[TorchAgent]
    agent_dict: Optional[DictionaryAgent]
    input_type: str
    generations: List[str]

    def tokenize_input(self, input: str) -> List[int]:
        """
        Tokenize input for the sub agent.

        Assumes that the sub agent has been instantiated.

        :param input:
            input to the sub agent

        :return tokens:
            return tokenized input
        """
        assert self.agents and self.agent_dict is not None
        return self.agent_dict.txt2vec(input)

    def clean_input(self, vec: torch.LongTensor) -> List[int]:
        """
        Clean a tensor before converting to a string.
        """
        assert self.agent_dict is not None
        return clean_vec_with_dict(self.agent_dict, vec)

    def _batch_generate(self, texts: List[str]) -> List[str]:
        """
        Batch generate items from an input list of texts.

        :param texts:
            list of texts

        :return generations:
            return agent generations for each input.
        """
        start = time.time()
        active_agents = self.agents[: len(texts)]
        for agent_i, t_i in zip(active_agents, texts):
            agent_i.observe(Message({'text': t_i, 'episode_done': True}))
        agent_replies = self.agents[0].batch_act([a.observation for a in active_agents])
        logging.debug(f'Generated: {time.time() - start:.2f}')
        for agent_i, reply_i in zip(active_agents, agent_replies):
            agent_i.self_observe(reply_i)
        self.generations = [r.get('text', 'dummy') for r in agent_replies]
        return self.generations


class QueryGenerator(BB2SubmoduleMixin):
    """
    The QueryGenerator is a wrapper around a generator model.

    This model can be trained for both dataset classification and search query
    generation.
    """

    def __init__(self, opt: Opt):
        self.opt = opt
        self.agents = []
        self.agent_dict = None
        self.generations = []
        self.input_type = 'Search'
        self.knowledge_access_method = KnowledgeAccessMethod(
            opt['knowledge_access_method']
        )
        model_file = modelzoo_path(opt['datapath'], opt['query_generator_model_file'])
        if (
            self.knowledge_access_method is KnowledgeAccessMethod.SEARCH_ONLY
            and 'blenderbot2/query_generator/model' in model_file
        ):
            raise ValueError(
                'You cannot use the blenderbot2 query generator with search_only. Please '
                'consider setting --query-generator-model-file zoo:sea/bart_sq_gen/model '
                'instead.'
            )
        if model_file and os.path.exists(model_file):
            logging.info(f'Building Query Generator from file: {model_file}')
            logging.disable()
            overrides: Dict[str, Any] = {'skip_generation': False}
            overrides['inference'] = opt['query_generator_inference']
            overrides['beam_size'] = opt.get('query_generator_beam_size', 3)
            overrides['beam_min_length'] = opt.get('query_generator_beam_min_length', 2)
            overrides['model_parallel'] = opt['model_parallel']
            overrides['no_cuda'] = opt['no_cuda']
            if self.opt['query_generator_truncate'] > 0:
                overrides['text_truncate'] = self.opt['query_generator_truncate']
                overrides['truncate'] = self.opt['query_generator_truncate']
            base_agent = create_agent_from_model_file(
                model_file, opt_overrides=overrides
            )
            assert isinstance(base_agent, TorchAgent)
            self.agents = [base_agent]
            bsz = max(opt.get('batchsize') or 1, opt.get('eval_batchsize') or 1)
            rag_turn_n_turns = opt.get('rag_turn_n_turns', 1)
            if bsz > 1 or rag_turn_n_turns > 1:
                self.agents += [
                    create_agent_from_shared(self.agents[0].share())
                    for _ in range((bsz * rag_turn_n_turns) - 1)
                ]
            self.agent_dict = self.agents[0].build_dictionary()
            logging.enable()

    def classify_retrieval(
        self,
        input: torch.LongTensor,
        num_memories: torch.LongTensor,
        generated_memories: Optional[List[List[str]]],
        skip_search: Optional[torch.BoolTensor],
    ) -> Tuple[torch.LongTensor, List[str]]:
        """
        Classify input and get retrieval type.

        Here, we classify which "type" of retrieval to do for each input batch item.

        In the case of "search", we additionally return search queries.

        :param input:
            input to classify
        :param num_memories:
            how many memories each example has.
            we override classification if there are no mems for the example.
        :param generated_memories:
            the generated memories from a memory decoder.

        :return (retrieval_type, searches):
            retrieval_type: a bsz-length tensor indicating which "type" of retrieval
                            we're doing (see RetrievalType above)
            searches: For batch items classified as search, we return the search queries
                      as well.
        """
        self.retrieval_type = torch.LongTensor(input.size(0))
        self.retrieval_type.fill_(0)
        assert self.agent_dict is not None
        texts = [self.agent_dict.vec2txt(self.clean_input(i)) for i in input]
        if self.knowledge_access_method is KnowledgeAccessMethod.MEMORY_ONLY:
            search_queries = [MEMORY_STRINGS[-1]] * len(texts)
        else:
            search_queries = self._batch_generate(texts)
        logging.debug(f'search queries: {search_queries}')
        logging.verbose(f'Search: {search_queries[0]}')
        searches = []
        if not generated_memories:
            generated_memories = [[] for _ in range(input.size(0))]
        for i, s in enumerate(search_queries):
            if (
                (strip_punc(s) in MEMORY_STRINGS)
                or any(ms in s for ms in MEMORY_STRINGS)
            ) and (
                (num_memories is not None and num_memories[i] > 0)
                or generated_memories[i]
            ):
                self.retrieval_type[i] = RetrievalType.MEMORY.value
            elif strip_punc(s) in NONE_STRINGS + MEMORY_STRINGS:
                self.retrieval_type[i] = RetrievalType.NONE.value
            elif skip_search is not None and skip_search[i]:
                self.retrieval_type[i] = RetrievalType.NONE.value
            else:
                self.retrieval_type[i] = RetrievalType.SEARCH.value
                searches.append(s)

        return self.retrieval_type, searches


class MemoryDecoder(BB2SubmoduleMixin):
    """
    Memory decoder.

    Given a line of context input, generate a memory to write.
    """

    def __init__(self, opt: Opt):
        self.opt = opt
        self.agents = []
        self.agent_dict = None
        self.generations = []
        self.input_type = 'Memory'
        self.delimiter = opt.get('memory_decoder_delimiter', '\n')
        self.one_line_memories = opt.get('memory_decoder_one_line_memories', False)
        model_file = modelzoo_path(opt['datapath'], opt['memory_decoder_model_file'])
        if model_file and os.path.exists(model_file):
            logging.info(f'Building Memory Decoder from file: {model_file}')
            logging.disable()
            overrides = {
                'skip_generation': False,
                'inference': 'beam',
                'beam_size': opt.get('memory_decoder_beam_size', 3),
                'beam_min_length': opt.get('memory_decoder_beam_min_length', 10),
                'beam_block_ngram': 3,
                'no_cuda': opt.get('no_cuda', False),
            }
            if self.opt.get('memory_decoder_truncate', -1) > 0:
                overrides['text_truncate'] = self.opt['memory_decoder_truncate']
                overrides['truncate'] = self.opt['memory_decoder_truncate']
            base_agent = create_agent_from_model_file(
                model_file, opt_overrides=overrides
            )
            assert isinstance(base_agent, TorchAgent)
            self.agents = [base_agent]
            assert isinstance(self.agents[0], TorchAgent)
            copies = max(100, (opt['batchsize'] * opt.get('rag_turn_n_turns', 1)))
            self.agents += [
                create_agent_from_shared(self.agents[0].share()) for _ in range(copies)
            ]
            self.agent_dict = self.agents[0].build_dictionary()
            logging.enable()

    def generate_memories(
        self, input: torch.LongTensor, num_inputs: torch.LongTensor
    ) -> List[List[str]]:
        """
        Generate memories from input.

        Each input is split into the lines of conversational context.
        These are considered independently.

        We then assign a prefix ("your/partner's persona:") dependent on
        whether the bot or it's partner said the line.

        :param input:
            input to the memory decoder
        :param num_inputs:
            number of lines per batch item
        """
        assert self.agent_dict is not None
        memories = []
        for idx, input_i in enumerate(input):
            if num_inputs[idx] == 0:
                continue
            context_lines_vec = input_i[: num_inputs[idx]]
            context_lines = [
                self.agent_dict.vec2txt(self.clean_input(j)) for j in context_lines_vec
            ]
            raw_memories_i = list(reversed(self._batch_generate(context_lines)))
            logging.debug(f'raw memories: {raw_memories_i}')
            memories_i = self._extract_from_raw_memories(raw_memories_i)
            logging.debug(f'memories to write: {memories_i}')
            mem_string = '\n'.join(memories_i)
            logging.verbose(f'Writing memories: {mem_string}')
            memories.append(memories_i)

        self.memories_full_list = memories
        return memories

    def _extract_from_raw_memories(self, raw_memories: List[str]) -> List[str]:
        """
        Extract memory lines from batch generated memories.

        Prefixes accordingly, and combines on one line if necessary.

        :param raw_memories:
            raw memory generations. sometimes we need skip the memories because
            nothing was generated

        :return memories:
            return prefixed and filtered memories
        """
        partner_prefix = 'partner\'s persona:'
        self_prefix = 'your persona:'
        num_ctxt = len(raw_memories)
        memories = []
        partner_memories = []
        self_memories = []
        for idx in range(num_ctxt):
            if raw_memories[idx] == NOPERSONA:
                continue
            if idx % 2 == 0:
                partner_memories.append(raw_memories[idx])
                prefix = partner_prefix
            else:
                self_memories.append(raw_memories[idx])
                prefix = self_prefix
            if not self.one_line_memories:
                memories.append(f'{prefix} {raw_memories[idx]}')
        if self.one_line_memories:
            if partner_memories:
                memories.append(f"{partner_prefix} {' '.join(partner_memories)}")
            if self_memories:
                memories.append(f"{self_prefix} {' '.join(self_memories)}")

        return memories
