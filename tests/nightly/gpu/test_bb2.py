#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import torch.cuda
import unittest
from parlai.core.agents import create_agent

import parlai.utils.testing as testing_utils

try:
    # blenderbot2 imports `transformer` and crashes the CPU tests.
    # These CPU tests will be skipped anyway with the decorators on each test.
    from projects.blenderbot2.agents.sub_modules import KnowledgeAccessMethod
    from projects.blenderbot2.agents.blenderbot2 import (
        ZOO_MEMORY_DECODER,
        ZOO_QUERY_GENERATOR,
    )

    TRANSFORMER_INSTALLED = True
except ImportError:
    TRANSFORMER_INSTALLED = False

LOCAL = True

if TRANSFORMER_INSTALLED:
    SEARCH_QUERY_MODEL = ZOO_QUERY_GENERATOR
    PERSONA_SUMMARY_MODEL = ZOO_MEMORY_DECODER
    ZOO_BB2 = 'zoo:blenderbot2/blenderbot2_400M/model'
    ZOO_BB2_3B = 'zoo:blenderbot2/blenderbot2_3B/model'
    SEARCH_SERVER = '<SERVER_API>'
    common_opt = {
        'model': 'projects.blenderbot2.agents.blenderbot2:BlenderBot2RagAgent',
        # rag args
        'init_opt': 'arch/bart_large',
        'generation_model': 'bart',
        'retriever_debug_index': 'compressed',
        'label_truncate': 128,
        'text_truncate': 512,
        'batchsize': 4,
        'fp16': True,
        'model_parallel': True,
        # train args
        'task': 'convai2,wizard_of_wikipedia',
        'num_examples': 8,
    }

    def _test_bb2_rag(retrieval_method: KnowledgeAccessMethod, **kwargs):
        opt = copy.deepcopy(common_opt)
        opt['knowledge_access_method'] = retrieval_method.value
        opt.update(dict(kwargs))
        testing_utils.eval_model(opt, skip_test=True)
        torch.cuda.empty_cache()

    def _test_bb2_fid(retrieval_method: KnowledgeAccessMethod, **kwargs):
        opt = copy.deepcopy(common_opt)
        opt['model'] = 'projects.blenderbot2.agents.blenderbot2:BlenderBot2FidAgent'
        opt['knowledge_access_method'] = retrieval_method.value
        opt.update(dict(kwargs))
        testing_utils.eval_model(opt, skip_test=True)
        torch.cuda.empty_cache()


@testing_utils.skipUnlessGPU
@unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
class TestBB2Rag(unittest.TestCase):
    """
    Test retrieval methods for BB2 with RAG.
    """

    def test_retrieval_all(self):
        _test_bb2_rag(KnowledgeAccessMethod.ALL)

    def test_retrieval_search_only(self):
        _test_bb2_rag(KnowledgeAccessMethod.SEARCH_ONLY)

    def test_retrieval_memory_only(self):
        _test_bb2_rag(KnowledgeAccessMethod.MEMORY_ONLY)

    def test_retrieval_classify(self):
        _test_bb2_rag(
            KnowledgeAccessMethod.CLASSIFY,
            query_generator_model_file=SEARCH_QUERY_MODEL,
        )

    def test_retrieval_none(self):
        _test_bb2_rag(KnowledgeAccessMethod.NONE, n_docs=1)


@testing_utils.skipIfCircleCI
class TestBB2Fid(unittest.TestCase):
    """
    Test retrieval methods for BB2 with FiD.
    """

    # BASIC Methods
    def test_retrieval_all(self):
        _test_bb2_fid(KnowledgeAccessMethod.ALL)

    def test_retrieval_search_only(self):
        _test_bb2_fid(KnowledgeAccessMethod.SEARCH_ONLY)

    def test_retrieval_memory_only(self):
        _test_bb2_fid(KnowledgeAccessMethod.MEMORY_ONLY)

    def test_retrieval_classify(self):
        _test_bb2_fid(
            KnowledgeAccessMethod.CLASSIFY,
            query_generator_model_file=SEARCH_QUERY_MODEL,
        )

    def test_retrieval_none(self):
        _test_bb2_fid(KnowledgeAccessMethod.NONE, n_docs=1)


@testing_utils.skipUnlessGPU
@unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
class TestLongTermMemory(unittest.TestCase):
    """
    Test LongTermMemory functionality.
    """

    def test_shared_encoders(self):
        _test_bb2_fid(
            KnowledgeAccessMethod.ALL, share_search_and_memory_query_encoder=True
        )


@testing_utils.skipUnlessGPU
@unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
class TestBB2RagTurn(unittest.TestCase):
    """
    Test RAG Turn functionality.
    """

    def test_rag_turn(self):
        _test_bb2_rag(
            KnowledgeAccessMethod.CLASSIFY,
            rag_model_type='turn',
            n_docs=3,
            batchsize=1,
            query_generator_model_file=SEARCH_QUERY_MODEL,
        )


@testing_utils.skipUnlessGPU
@unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
class TestBB2Search(unittest.TestCase):
    """
    Test Search functionality.
    """

    def test_rag(self):
        _test_bb2_rag(
            KnowledgeAccessMethod.CLASSIFY,
            rag_model_type='token',
            n_docs=3,
            batchsize=1,
            query_generator_model_file=SEARCH_QUERY_MODEL,
            rag_retriever_type='search_engine',
            search_server=SEARCH_SERVER,
        )


@testing_utils.skipUnlessGPU
@unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
class TestBB2RagSequence(unittest.TestCase):
    """
    Test RAG Sequence functionality.
    """

    def test_rag(self):
        _test_bb2_rag(
            KnowledgeAccessMethod.CLASSIFY,
            rag_model_type='sequence',
            n_docs=3,
            batchsize=1,
            query_generator_model_file=SEARCH_QUERY_MODEL,
            rag_retriever_type='search_engine',
            search_server=SEARCH_SERVER,
        )


@testing_utils.skipUnlessGPU
@unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
class TestBB2QGenParams(unittest.TestCase):
    """
    Test RAG Turn functionality.
    """

    def test_rag(self):
        _test_bb2_rag(
            KnowledgeAccessMethod.CLASSIFY,
            rag_model_type='token',
            n_docs=3,
            batchsize=1,
            query_generator_model_file=SEARCH_QUERY_MODEL,
            rag_retriever_type='search_engine',
            search_server=SEARCH_SERVER,
            query_generator_beam_size=3,
            query_generator_beam_min_length=2,
        )


@testing_utils.skipUnlessGPU
@unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
class TestBB2CleanText(unittest.TestCase):
    SPEICIAL_TOKEN = '_POTENTIALLY_UNSAFE__'

    def test_bb2_history(self):
        """
        Test out-of-the-box BB2 generation.
        """
        opt = copy.deepcopy(common_opt)
        opt.update(
            {
                'model_file': ZOO_BB2,
                'override': {
                    'search_server': SEARCH_SERVER,
                    'add_cleaned_reply_to_history': True,
                },
            }
        )
        bb2 = create_agent(opt)

        text_with_safety_token = f"Don't have a cow, Man! {self.SPEICIAL_TOKEN}"
        obs = {'text': text_with_safety_token}
        bb2.observe(obs)
        assert self.SPEICIAL_TOKEN in bb2.history.get_history_str()

        bb2.history.reset()
        obs = {'text': "I am Groot"}
        bb2.observe(obs)
        bb2.history.add_reply(text_with_safety_token)
        assert self.SPEICIAL_TOKEN not in bb2.history.get_history_str()


@testing_utils.skipUnlessGPU
@unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
class TestBB2AdditionalTruncation(unittest.TestCase):
    """
    Test RAG Turn functionality.
    """

    def test_rag(self):
        _test_bb2_rag(
            KnowledgeAccessMethod.CLASSIFY,
            rag_model_type='token',
            n_docs=3,
            batchsize=1,
            query_generator_model_file=SEARCH_QUERY_MODEL,
            rag_retriever_type='search_engine',
            search_server=SEARCH_SERVER,
            query_generator_truncate=24,
            memory_retriever_truncate=24,
        )


@testing_utils.skipUnlessGPU
@unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
class TestBB2GoldDocs(unittest.TestCase):
    """
    Test RAG Turn functionality.
    """

    def test_rag(self):
        _test_bb2_rag(
            KnowledgeAccessMethod.CLASSIFY,
            rag_model_type='sequence',
            n_docs=3,
            batchsize=1,
            query_generator_model_file=SEARCH_QUERY_MODEL,
            rag_retriever_type='search_engine',
            search_server=SEARCH_SERVER,
            insert_gold_docs=True,
            task='wizard_of_internet',
        )


@testing_utils.skipUnlessGPU
@unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
class TestBB2MemoryDecoder(unittest.TestCase):
    """
    Test RAG Turn functionality.
    """

    def test_rag(self):
        _test_bb2_rag(
            KnowledgeAccessMethod.CLASSIFY,
            rag_model_type='sequence',
            n_docs=3,
            batchsize=1,
            query_generator_model_file=SEARCH_QUERY_MODEL,
            rag_retriever_type='search_engine',
            search_server=SEARCH_SERVER,
            insert_gold_docs=True,
            task='wizard_of_internet',
            memory_decoder_model_file=PERSONA_SUMMARY_MODEL,
        )


@unittest.skipUnless(TRANSFORMER_INSTALLED, "Needs transformer, not installed.")
class TestBB2ZooModel(unittest.TestCase):
    """
    Test Zoo Model.
    """

    def test_zoo_model(self):
        _test_bb2_fid(
            KnowledgeAccessMethod.CLASSIFY,
            n_docs=3,
            batchsize=1,
            task='wizard_of_internet',
            model_file=ZOO_BB2,
            rag_retriever_type='dpr',
            indexer_type='compressed',
        )

    @unittest.skipIf(LOCAL, "Skipping Test because its slow and mem intensive")
    def test_zoo_model_3B(self):
        _test_bb2_fid(
            KnowledgeAccessMethod.CLASSIFY,
            n_docs=3,
            batchsize=1,
            task='wizard_of_internet',
            model_file=ZOO_BB2_3B,
            search_server=SEARCH_SERVER,
            init_opt='gen/blenderbot',
            generation_model='transformer/generator',
        )


if __name__ == '__main__':
    unittest.main()
