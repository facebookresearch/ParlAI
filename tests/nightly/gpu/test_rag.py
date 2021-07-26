#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import torch
import torch.cuda
from typing import Optional
import unittest

from parlai.core.build_data import modelzoo_path
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser, Opt
import parlai.utils.testing as testing_utils

try:
    from parlai.agents.rag.dpr import DprQueryEncoder
except ImportError:
    pass

from parlai.agents.rag.args import (
    DPR_ZOO_MODEL,
    POLYFAISS_ZOO_MODEL,
    RAG_TOKEN_ZOO_MODEL,
    RAG_SEQUENCE_ZOO_MODEL,
    RAG_TURN_DO_ZOO_MODEL,
    RAG_TURN_DTT_ZOO_MODEL,
    RAG_DPR_POLY_ZOO_MODEL,
    FID_DPR_ZOO_MODEL,
    FID_RAG_ZOO_MODEL,
    FID_RAG_DPR_POLY_ZOO_MODEL,
)

common_opt = {
    'model': 'rag',
    'retriever_debug_index': 'compressed',
    'dpr_model_file': DPR_ZOO_MODEL,
    'n_docs': 2,
    'task': 'integration_tests',
    'num_examples': 1,
    'label_truncate': 5,
    'indexer_type': 'compressed',
    'compressed_indexer_gpu_train': False,
    'rag_turn_n_turns': 2,
}

test_opt = {
    **common_opt,
    'init_model': 'zoo:unittest/transformer_generator2/model',
    'dict_file': 'zoo:unittest/transformer_generator2/model.dict',
    'n_layers': 2,
    'n_heads': 2,
    'embedding_size': 32,
    'ffn_size': 128,
    'n_positions': 1024,
    'dict_tokenizer': 're',
    'generation_model': 'transformer/generator',
}

rag_dpr_model_file = RAG_TOKEN_ZOO_MODEL
polyfaiss_model_file = POLYFAISS_ZOO_MODEL

GENERATION_OPTIONS = ['bart', 't5', 'transformer/generator']
GENERATION_OPTS = {
    'bart': {
        'embedding_size': 1024,
        'ffn_size': 4096,
        'n_layers': 12,
        'n_heads': 16,
        'n_positions': 1024,
        'variant': 'bart',
        'truncate': 64,
        'dict_tokenizer': 'gpt2',
        'init_model': 'zoo:bart/bart_large/model',
        'dict_file': 'zoo:bart/bart_large/model.dict',
        'fp16': True,
    },
    't5': {'t5_model_arch': 't5-small', 'fp16': True, 'truncate': 64},
    'transformer/generator': {
        'init_model': 'zoo:tutorial_transformer_generator/model',
        'dict_file': 'zoo:tutorial_transformer_generator/model.dict',
        'dict_tokenizer': 'bpe',
        'variant': 'xlm',
        'activation': 'gelu',
        'text_truncate': 64,
        'label_truncate': 64,
        'n_positions': 512,
        'ffn_size': 2048,
        'n_layers': 8,
        'n_encoder_layers': 8,
        'n_decoder_layers': 8,
        'embedding_size': 512,
        'n_heads': 16,
        'fp16': True,
    },
}

RAG_MODEL_TYPES = ['token', 'sequence', 'turn']
RAG_TOKEN_OPTIONS = {}
RAG_SEQUENCE_OPTIONS = {}
RAG_MODEL_TYPE_OPTIONS = {
    'token': {'thorough': [False]},
    'sequence': {'thorough': [False, True]},
    'turn': {'rag_turn_marginalize': ['doc_then_turn', 'doc_only']},
    'turn:thorough=True': {'rag_turn_marginalize': ['doc_only']},
}


@testing_utils.skipUnlessGPU
class TestRagDpr(unittest.TestCase):
    """
    Test all RAG DPR Model Types with Base Generators.
    """

    def _test_rag_type(self, model_type: str, gen_model: str, no_cuda: bool = False):
        opt = copy.deepcopy(common_opt)
        opt['generation_model'] = gen_model
        opt.update(GENERATION_OPTS[gen_model])
        opt['rag_model_type'] = model_type.split(':')[0]
        opt['no_cuda'] = no_cuda
        for vals in model_type.split(':'):
            if '=' in vals:
                k, v = vals.split('=')
                opt[k] = bool(v)
        for option, vals in RAG_MODEL_TYPE_OPTIONS[model_type].items():
            for val in vals:
                opt[option] = val
                testing_utils.eval_model(opt, skip_test=True)

    def test_bart_rag_sequence(self):
        self._test_rag_type('sequence', 'bart')

    def test_bart_rag_token(self):
        self._test_rag_type('token', 'bart', no_cuda=True)

    def test_bart_rag_turn(self):
        self._test_rag_type('turn', 'bart', no_cuda=True)

    def test_bart_rag_turn_thorough(self):
        self._test_rag_type('turn:thorough=True', 'bart', no_cuda=True)

    def test_t5_rag_sequence(self):
        self._test_rag_type('sequence', 't5', no_cuda=True)

    def test_t5_rag_token(self):
        self._test_rag_type('token', 't5')

    def test_t5_rag_turn(self):
        self._test_rag_type('turn', 't5', no_cuda=True)

    def test_t5_rag_turn_thorough(self):
        self._test_rag_type('turn:thorough=True', 't5', no_cuda=True)

    def test_reddit_rag_sequence(self):
        self._test_rag_type('sequence', 'transformer/generator', no_cuda=True)

    def test_reddit_rag_token(self):
        self._test_rag_type('token', 'transformer/generator', no_cuda=True)

    def test_reddit_rag_turn(self):
        self._test_rag_type('turn', 'transformer/generator')

    def test_reddit_rag_turn_thorough(self):
        self._test_rag_type('turn:thorough=True', 'transformer/generator', no_cuda=True)


@testing_utils.skipUnlessGPU
class TestFidDpr(unittest.TestCase):
    """
    Test FiD DPR Model.
    """

    def _test_fid(self, gen_model: str, no_cuda: bool = False):
        opt = copy.deepcopy(common_opt)
        opt['model'] = 'fid'
        opt['generation_model'] = gen_model
        opt['no_cuda'] = no_cuda
        opt.update(GENERATION_OPTS[gen_model])
        testing_utils.eval_model(opt, skip_test=True)

    def test_bart_fid(self):
        self._test_fid('bart', no_cuda=True)

    def test_t5_fid(self):
        self._test_fid('t5', no_cuda=True)

    def test_reddit_fid(self):
        self._test_fid('transformer/generator')


@testing_utils.skipUnlessGPU
class TestRagDprPoly(unittest.TestCase):
    """
    Test RAG DPR Poly model.
    """

    def _test_rag_type(self, model_type: str, no_cuda: bool = False):
        opt = copy.deepcopy(test_opt)
        opt['rag_retriever_type'] = 'dpr_then_poly'
        opt['rag_model_type'] = model_type
        opt['no_cuda'] = no_cuda
        for option, vals in RAG_MODEL_TYPE_OPTIONS[model_type].items():
            for val in vals:
                opt[option] = val
                testing_utils.eval_model(opt, skip_test=True)

    def test_rag_sequence(self):
        self._test_rag_type('sequence', no_cuda=True)

    def test_rag_token(self):
        self._test_rag_type('token')

    def test_rag_turn(self):
        self._test_rag_type('turn', no_cuda=True)


@testing_utils.skipUnlessGPU
class TestRagTfidf(unittest.TestCase):
    """
    Test RAG TFIDF model.
    """

    def test_rag_token(self):
        opt = copy.deepcopy(test_opt)
        opt['rag_retriever_type'] = 'tfidf'
        opt['rag_model_type'] = 'token'
        testing_utils.eval_model(opt, skip_test=True)


@testing_utils.skipUnlessGPU
class TestFidRag(unittest.TestCase):
    """
    Test Fid Rag.
    """

    def _test_fid(self, gen_model: str, no_cuda: bool = False):
        opt = copy.deepcopy(common_opt)
        opt['generation_model'] = gen_model
        opt['model'] = 'fid'
        opt['query_model'] = 'bert_from_parlai_rag'
        opt['dpr_model_file'] = rag_dpr_model_file
        opt['no_cuda'] = no_cuda
        opt.update(GENERATION_OPTS[gen_model])
        testing_utils.eval_model(opt, skip_test=True)

    def test_bart_fid(self):
        self._test_fid('bart', no_cuda=True)

    def test_t5_fid(self):
        self._test_fid('t5', no_cuda=True)

    def test_reddit_fid(self):
        self._test_fid('transformer/generator')


@testing_utils.skipUnlessGPU
class TestRagPolyfaiss(unittest.TestCase):
    """
    Test Rag PolyFAISS.
    """

    def test_bart_rag_token(self):
        opt = copy.deepcopy(test_opt)
        opt['query_model'] = 'dropout_poly'
        opt['rag_retriever_type'] = 'poly_faiss'
        opt['poly_faiss_model_file'] = polyfaiss_model_file
        opt['rag_model_type'] = 'token'
        testing_utils.eval_model(opt, skip_test=True)


@testing_utils.skipUnlessGPU
class TestRegret(unittest.TestCase):
    """
    Test ReGReT.
    """

    def _test_regret(self, regret_mf: Optional[str] = None):
        opt = copy.deepcopy(test_opt)
        opt['regret_model_file'] = regret_mf
        opt['rag_model_type'] = 'token'
        opt['no_cuda'] = True
        testing_utils.eval_model(opt, skip_test=True)

    def test_rag_regret_sep(self):
        self._test_regret(RAG_TOKEN_ZOO_MODEL)

    def test_rag_regret_same(self):
        self._test_regret()


@testing_utils.skipUnlessGPU
class TestOtherOptions(unittest.TestCase):
    """
    Test other RAG Options.
    """

    def test_n_positions(self):
        opt = copy.deepcopy(test_opt)
        opt['rag_model_type'] = 'token'
        opt['n_extra_positions'] = 128
        testing_utils.eval_model(opt, skip_test=True)

    def test_resize_embs(self):
        opt = copy.deepcopy(test_opt)
        opt['rag_model_type'] = 'token'
        opt['special_tok_lst'] = '__hello__,__goodbye__'
        testing_utils.eval_model(opt, skip_test=True)


@testing_utils.skipUnlessGPU
class TestQueryModels(unittest.TestCase):
    """
    Test other RAG Options.
    """

    def test_dpr_agent(self):
        # eval only
        opt = {
            'task': 'integration_tests:overfit',
            'model': 'dpr_agent',
            'model_file': DPR_ZOO_MODEL,
            'num_examples': 5,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_dropout_poly(self):
        opt = {
            'task': 'integration_tests:overfit',
            'model': 'transformer/dropout_poly',
            'optimizer': 'adam',
            'learningrate': 1e-2,
            'batchsize': 4,
            'validation_every_n_epochs': 5,
            'validation_patience': 10,
            'lr_scheduler': 'none',
            'embedding_size': 8,
            'gradient_clip': 0.5,
            'n_layers': 1,
            'n_heads': 4,
            'ffn_size': 32,
        }
        valid, _ = testing_utils.train_model(opt)
        assert float(valid['accuracy']) >= 0.6


def _test_zoo_file(mf: str, fid: bool = False, fid_rag: bool = False):
    opt = copy.deepcopy(common_opt)
    if fid:
        opt['model'] = 'fid'
    if fid_rag:
        opt['dpr_model_file'] = RAG_TOKEN_ZOO_MODEL
    opt.update(GENERATION_OPTS['bart'])
    opt['model_file'] = mf
    opt['generation_model'] = 'bart'
    opt['task'] = 'wizard_of_wikipedia'
    opt['label_truncate'] = 10
    valid, _ = testing_utils.eval_model(opt, skip_test=True)
    assert valid['ppl'] < 15.0
    assert (100 * float(valid['f1'])) > 10.0
    torch.cuda.empty_cache()


@testing_utils.skipUnlessGPU
class TestRagZooModels(unittest.TestCase):
    """
    Test ZOO Models.
    """

    def test_bart_rag_token(self):
        _test_zoo_file(RAG_TOKEN_ZOO_MODEL)

    def test_bart_rag_sequence(self):
        _test_zoo_file(RAG_SEQUENCE_ZOO_MODEL)

    def test_bart_rag_dpr_poly(self):
        _test_zoo_file(RAG_DPR_POLY_ZOO_MODEL)

    def test_bart_rag_turn_dtt(self):
        _test_zoo_file(RAG_TURN_DTT_ZOO_MODEL)

    def test_bart_rag_turn_do(self):
        _test_zoo_file(RAG_TURN_DO_ZOO_MODEL)


@testing_utils.skipUnlessGPU
class TestFidZooModels(unittest.TestCase):
    """
    Test FiD zoo models.
    """

    def test_bart_fid_dpr(self):
        _test_zoo_file(FID_DPR_ZOO_MODEL, True)

    def test_bart_fid_rag(self):
        _test_zoo_file(FID_RAG_ZOO_MODEL, True, True)

    def test_bart_fid_rag_dpr_poly(self):
        _test_zoo_file(FID_RAG_DPR_POLY_ZOO_MODEL, True, True)


class TestLoadDPRModel(unittest.TestCase):
    """
    Test loading different DPR models for RAG.

    Suppose we have the following models:

    1. A: Default DPR Model
    2. M: RAG Model trained with A
    3. B: Resulting DPR Model after training M
    4. C: DPR Model from training a different RAG Model

    The following should hold true:

    1. `parlai em -mf M` -> M.DPR (B) != A
    2. `parlai em -mf M --dpr-model-file A` -> M.DPR == A
    3. `parlai em -mf M --dpr-model-file C` -> M.DPR (C) != B
    """

    def test_load_dpr(self):
        opt = ParlaiParser(True, True).parse_args([])
        # First, we'll load up a DPR model from the zoo dpr file.
        default_query_encoder = DprQueryEncoder(
            opt, dpr_model='bert', pretrained_path=DPR_ZOO_MODEL
        )
        rag_sequence_query_encoder = DprQueryEncoder(
            opt,
            dpr_model='bert_from_parlai_rag',
            pretrained_path=RAG_SEQUENCE_ZOO_MODEL,
        )
        assert not torch.allclose(
            default_query_encoder.embeddings.weight.float().cpu(),
            rag_sequence_query_encoder.embeddings.weight.float().cpu(),
        )
        # 1. Create a zoo RAG Agent, which involves a trained DPR model
        rag = create_agent(
            Opt(
                {
                    'model_file': modelzoo_path(opt['datapath'], RAG_TOKEN_ZOO_MODEL),
                    'override': {'retriever_debug_index': 'compressed', 'fp16': False},
                }
            )
        )
        # The default rag token model should have different query encoders
        # from both the RAG_SEQUENCE_ZOO_MODEL, and the default DPR_ZOO_MODEL
        assert not torch.allclose(
            rag_sequence_query_encoder.embeddings.weight.float().cpu(),
            rag.model.retriever.query_encoder.embeddings.weight.float().cpu(),
        )
        assert not torch.allclose(
            default_query_encoder.embeddings.weight.float().cpu(),
            rag.model.retriever.query_encoder.embeddings.weight.float().cpu(),
        )

        # 2. create a RAG Agent with the rag_sequence_zoo_model DPR model
        rag = create_agent(
            Opt(
                {
                    'model_file': modelzoo_path(opt['datapath'], RAG_TOKEN_ZOO_MODEL),
                    'override': {
                        'retriever_debug_index': 'compressed',
                        'dpr_model_file': modelzoo_path(
                            opt['datapath'], RAG_SEQUENCE_ZOO_MODEL
                        ),
                        'query_model': 'bert_from_parlai_rag',
                        'fp16': False,
                    },
                }
            )
        )
        # If we override the DPR Model file, we should now have the same
        # weights as the query encoder from above.
        assert torch.allclose(
            rag_sequence_query_encoder.embeddings.weight.float().cpu(),
            rag.model.retriever.query_encoder.embeddings.weight.float().cpu(),
        )

        # 3. Create a RAG Agent with the default DPR zoo model
        rag = create_agent(
            Opt(
                {
                    'model_file': modelzoo_path(opt['datapath'], RAG_TOKEN_ZOO_MODEL),
                    'override': {
                        'retriever_debug_index': 'compressed',
                        'dpr_model_file': modelzoo_path(opt['datapath'], DPR_ZOO_MODEL),
                        'fp16': False,
                    },
                }
            )
        )

        # This model was trained with the DPR_ZOO_MODEL, and yet now should have the same weights
        # as we explicitly specified it.
        assert torch.allclose(
            default_query_encoder.embeddings.weight.float().cpu(),
            rag.model.retriever.query_encoder.embeddings.weight.float().cpu(),
        )


if __name__ == '__main__':
    unittest.main()
