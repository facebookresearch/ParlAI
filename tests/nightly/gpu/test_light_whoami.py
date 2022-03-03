#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest

from parlai.core.message import Message
from parlai.core.params import ParlaiParser, Opt
from parlai.core.torch_ranker_agent import TorchRankerAgent
import parlai.utils.testing as testing_utils

from projects.light_whoami.agents import (
    RPA_RERANKER,
    RPA_RERANKER_AUTO_EXPANDED,
    VANILLA_128,
    VANILLA_1024,
    UL_128,
    UL_1024,
    MULTIOBJECTIVE,
    PROFILE_EXPANDED_128,
    PROFILE_EXPANDED_1024,
    AUTO_EXPANDED_1024,
    EXPANDED_AND_MULTIOBJECTIVE_1024,
    TEST_RPA_RERANKER,
)

LOCAL_TEST = False

TEST_TGA = 'zoo:blender/blender_90M/model'


LIGHT_EXAMPLE = Message(
    {
        'text': '_setting_name Turquoise Shore, Shore\n'
        '_setting_desc A beautiful turquoise color water by the shore. It is filled with many gems and gold.\n'
        '_partner_name mermaid\n'
        '_self_name sea witch\n'
        '_self_persona I am a sea witch.  I pray on young sailors who hope to find adventure and treasures on the open sea.  I lure them in with magic spells and promise of riches.\n'
        'Hey there Mermaid! Long time, no see.',
        'labels': ['Long time indeed! How have you been keeping?'],
    }
)


COMMON_OPT = {
    'task': 'projects.light_whoami.task.agents:BaseSimpleMultiTeacher',
    'num_examples': 2,
    'skip_generation': False,
    'inference': 'beam',
    'beam_size': 3,
    'beam_block_full_context': True,
}

TRAIN_COMMON_OPT = {
    'max_train_steps': 1,
    'validation_max_exs': 5,
    'short_final_eval': True,
    'truncate': 16,
    'text_truncate': 16,
    'label_truncate': 16,
    'init_model': TEST_TGA,
    'dict_file': f"{TEST_TGA}.dict",
    'n_layers': 8,
    'n_heads': 16,
    'embedding_size': 512,
    'ffn_size': 2048,
    'n_positions': 512,
    'dict_tokenizer': 'bpe',
    'optimizer': 'sgd',
    'variant': 'xlm',
    'skip_generation': True,
}

TEST_RERANKER_OPTS = {
    'dict_tokenizer': 'bytelevelbpe',
    'dict_file': f"{TEST_RPA_RERANKER}.dict",
    'bpe_merge': f"{TEST_RPA_RERANKER}.dict-merges.txt",
    'bpe_vocab': f"{TEST_RPA_RERANKER}.dict-vocab.json",
}


@testing_utils.skipUnlessGPU
class TestReranker(unittest.TestCase):
    """
    Test Re-ranker Functionality.
    """

    def _setup_parser(self) -> Opt:
        from projects.light_whoami.agents.rpa_rerank import RPAReranker

        parser = ParlaiParser(True, True)
        parser = RPAReranker.add_cmdline_args(parser, {})
        parser = TorchRankerAgent.add_cmdline_args(parser, {})
        opt = parser.parse_args(['--predictor-model-file', RPA_RERANKER])
        return opt

    def test_light_whoami_reranker(self):
        """
        Test re-ranker.
        """
        from projects.light_whoami.agents.rpa_rerank import RPAReranker

        opt = self._setup_parser()
        reranker = RPAReranker(opt)

        # testing abstract function impl.
        assert (
            reranker.get_class_to_rerank_for({}, LIGHT_EXAMPLE['text']) == 'sea witch'
        )
        assert all(
            reranker.is_context(c) for c in LIGHT_EXAMPLE['text'].split('\n')[:-1]
        )
        assert reranker.get_predictor_label_candidates({}, LIGHT_EXAMPLE['text']) == [
            'sea witch',
            'mermaid',
        ]

        output = reranker.classify(LIGHT_EXAMPLE['text'], LIGHT_EXAMPLE['labels'][0])
        assert output['text_candidates'] == ['sea witch', 'mermaid']
        assert output['text'] == 'sea witch'


@testing_utils.skipUnlessGPU
class TestGenerativeRerank(unittest.TestCase):
    """
    Test Generative Re-rankers.
    """

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_short_rerank_pretrain(self):
        opt = {
            'model': 'projects.light_whoami.agents.rpa_rerank:RPARerankAgent',
            'model_file': VANILLA_128,
            'predictor_model_file': RPA_RERANKER,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_long_rerank_pretrain(self):
        opt = {
            'model': 'projects.light_whoami.agents.rpa_rerank:LongRPARerankAgent',
            'model_file': VANILLA_1024,
            'predictor_model_file': RPA_RERANKER,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_short_rerank(self):
        opt = {
            'model': 'projects.light_whoami.agents.rpa_rerank:RPARerankAgent',
            'model_file': TEST_TGA,
            'predictor_model_file': RPA_RERANKER,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_long_rerank(self):
        opt = {
            'model': 'projects.light_whoami.agents.rpa_rerank:LongRPARerankAgent',
            'model_file': TEST_TGA,
            'predictor_model_file': RPA_RERANKER,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)


@testing_utils.skipUnlessGPU
class TestPacer(unittest.TestCase):
    """
    Test Pacer Agents.
    """

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_short_pacer_pretrain(self):
        opt = {
            'model': 'projects.light_whoami.agents.pacer:PacerPartialOnlyAgent',
            'model_file': VANILLA_128,
            'predictor_model_file': RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)
        opt = {
            'model': 'projects.light_whoami.agents.pacer:PacerAgent',
            'model_file': VANILLA_128,
            'predictor_model_file': RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_long_pacer_pretrain(self):
        opt = {
            'model': 'projects.light_whoami.agents.pacer:LongPacerPartialOnlyAgent',
            'model_file': VANILLA_1024,
            'predictor_model_file': RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)
        opt = {
            'model': 'projects.light_whoami.agents.pacer:LongPacerAgent',
            'model_file': VANILLA_1024,
            'predictor_model_file': RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_short_pacer(self):
        opt = {
            'model': 'projects.light_whoami.agents.pacer:PacerPartialOnlyAgent',
            'model_file': TEST_TGA,
            'predictor_model_file': TEST_RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)
        opt = {
            'model': 'projects.light_whoami.agents.pacer:PacerAgent',
            'model_file': TEST_TGA,
            'predictor_model_file': TEST_RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_long_pacer(self):
        opt = {
            'model': 'projects.light_whoami.agents.pacer:LongPacerPartialOnlyAgent',
            'model_file': TEST_TGA,
            'predictor_model_file': TEST_RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)
        opt = {
            'model': 'projects.light_whoami.agents.pacer:LongPacerAgent',
            'model_file': TEST_TGA,
            'predictor_model_file': TEST_RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)


@testing_utils.skipUnlessGPU
class TestRpaUnlikelihood(unittest.TestCase):
    """
    Test Generative Re-rankers.
    """

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_short_pretrained(self):
        opt = {
            'model': 'projects.light_whoami.agents.rpa_ul:RpaUlAgent',
            'model_file': UL_128,
            'predictor_model_file': RPA_RERANKER,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_short_training(self):
        opt = {
            'model': 'projects.light_whoami.agents.rpa_ul:RpaUlAgent',
            'predictor_model_file': TEST_RPA_RERANKER,
            **COMMON_OPT,
        }
        opt.pop('num_examples')
        opt.update(TRAIN_COMMON_OPT)
        testing_utils.train_model(opt)

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_long_pretrained(self):
        opt = {
            'model': 'projects.light_whoami.agents.rpa_ul:LongRpaUlAgent',
            'model_file': UL_1024,
            'predictor_model_file': RPA_RERANKER,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)


@testing_utils.skipUnlessGPU
class TestMultiobjective(unittest.TestCase):
    """
    Test multiobjective models.
    """

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_short_multiobj_pretrained(self):
        opt = {
            'model': 'projects.light_whoami.agents.multi_objective:MultiObjectiveGeneratorAgent',
            'model_file': MULTIOBJECTIVE,
            **COMMON_OPT,
            'task': 'projects.light_whoami.task.agents:MultiObjectiveTeacher',
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_short_multiobj_training(self):
        opt = {
            'model': 'projects.light_whoami.agents.multi_objective:MultiObjectiveGeneratorAgent',
            'n_multiobjective_heads': 4,
            'n_multiobjective_layers': 2,
            **COMMON_OPT,
            'task': 'projects.light_whoami.task.agents:MultiObjectiveTeacher',
        }
        opt.pop('num_examples')
        opt.update(TRAIN_COMMON_OPT)
        for choice in [
            'decoder_final_layer',
            'encoder_final_layer',
            'encoder_and_decoder',
        ]:
            opt['multiobjective_latent_representation'] = choice
            testing_utils.train_model(opt)


@testing_utils.skipUnlessGPU
class TestExpandedAttentionProfile(unittest.TestCase):
    """
    Test multiobjective models.
    """

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_short_exp_attn(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAgent',
            'model_file': PROFILE_EXPANDED_128,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_short_exp_attention_train(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAgent',
            'expanded_attention_share_weights': True,
            'expanded_attention_input_extractor_phrases': "_self_name,_self_persona,_partner_name",
            'expanded_attention_type': 'profile',
            'expanded_attention_num_rounds': 2,
            **COMMON_OPT,
        }
        opt.pop('num_examples')
        opt.update(TRAIN_COMMON_OPT)
        testing_utils.train_model(opt)

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_long_exp_attn(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:LongExpandedDecoderAttentionAgent',
            'model_file': PROFILE_EXPANDED_1024,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_long_exp_attn_train(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:LongExpandedDecoderAttentionAgent',
            'expanded_attention_share_weights': True,
            'expanded_attention_input_extractor_phrases': "_self_name,_self_persona,_partner_name,_setting_name,_setting_desc",
            'expanded_attention_type': 'profile',
            'expanded_attention_num_rounds': 2,
            **COMMON_OPT,
        }
        opt.pop('num_examples')
        opt.update(TRAIN_COMMON_OPT)
        testing_utils.train_model(opt)


@testing_utils.skipUnlessGPU
class TestExpandedAttentionAutomated(unittest.TestCase):
    """
    Test multiobjective models.
    """

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_long_exp_attn(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:LongExpandedDecoderAttentionAgent',
            'model_file': AUTO_EXPANDED_1024,
            'mutators': 'clean_context_mutator,share_self_character',
            'expanded_attention_classifier_model_file': RPA_RERANKER_AUTO_EXPANDED,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_exp_attn_train_automated(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAgent',
            'mutators': 'clean_context_mutator,share_self_character',
            'expanded_attention_share_weights': True,
            'expanded_attention_type': 'automated_classifier',
            'automated_expanded_attention_n_tokens': 10,
            'expanded_attention_classifier_model_file': TEST_RPA_RERANKER,
            **COMMON_OPT,
        }
        opt.update(TRAIN_COMMON_OPT)
        opt.pop('num_examples')
        opt.pop('init_model')
        opt.pop('dict_file')
        opt.update(TEST_RERANKER_OPTS)
        testing_utils.train_model(opt)


@testing_utils.skipUnlessGPU
class TestExpandedAttentionAndReranker(unittest.TestCase):
    """
    Test Generative Re-rankers.
    """

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_short_rerank_pretrain(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAndRPARerankerAgent',
            'model_file': PROFILE_EXPANDED_128,
            'predictor_model_file': RPA_RERANKER,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_short_rerank(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAndRPARerankerAgent',
            'model_file': TEST_TGA,
            'predictor_model_file': TEST_RPA_RERANKER,
            'expanded_attention_share_weights': True,
            'expanded_attention_input_extractor_phrases': "_self_name,_self_persona,_partner_name",
            'expanded_attention_type': 'profile',
            'expanded_attention_num_rounds': 2,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_short_pacer_pretrain(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAndPacerAgent',
            'model_file': PROFILE_EXPANDED_128,
            'predictor_model_file': RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_short_pacer(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAndPacerAgent',
            'model_file': TEST_TGA,
            'predictor_model_file': TEST_RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            'expanded_attention_share_weights': True,
            'expanded_attention_input_extractor_phrases': "_self_name,_self_persona,_partner_name",
            'expanded_attention_type': 'profile',
            'expanded_attention_num_rounds': 2,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_long_rerank_pretrain(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:LongExpandedDecoderAttentionAndRPARerankerAgent',
            'model_file': PROFILE_EXPANDED_1024,
            'predictor_model_file': RPA_RERANKER,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_long_rerank(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:LongExpandedDecoderAttentionAndRPARerankerAgent',
            'model_file': TEST_TGA,
            'predictor_model_file': TEST_RPA_RERANKER,
            'expanded_attention_share_weights': True,
            'expanded_attention_input_extractor_phrases': "_self_name,_self_persona,_partner_name,_setting_name,_setting_desc",
            'expanded_attention_type': 'profile',
            'expanded_attention_num_rounds': 2,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_long_pacer_pretrain(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:LongExpandedDecoderAttentionAndPacerAgent',
            'model_file': PROFILE_EXPANDED_1024,
            'predictor_model_file': RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_long_pacer(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:LongExpandedDecoderAttentionAndPacerAgent',
            'model_file': TEST_TGA,
            'predictor_model_file': TEST_RPA_RERANKER,
            'pacer_n_tokens': 10,
            'pacer_frequency_ratio': 0.1,
            'beam_min_length': 10,
            'expanded_attention_share_weights': True,
            'expanded_attention_input_extractor_phrases': "_self_name,_self_persona,_partner_name,_setting_name,_setting_desc",
            'expanded_attention_type': 'profile',
            'expanded_attention_num_rounds': 2,
            **COMMON_OPT,
        }
        testing_utils.eval_model(opt, skip_test=True)


@testing_utils.skipUnlessGPU
class TestExpandedAttentionAutomatedAndMultiObj(unittest.TestCase):
    """
    Test multiobjective models.
    """

    @unittest.skipUnless(LOCAL_TEST, 'Skipping due to CI Memory Constraints')
    def test_long_exp_attn(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:LongExpandedDecoderAttentionAndMultiObjectiveAgent',
            'model_file': EXPANDED_AND_MULTIOBJECTIVE_1024,
            'mutators': 'clean_context_mutator,share_self_character',
            **COMMON_OPT,
            'task': 'projects.light_whoami.task.agents:MultiObjectiveTeacher',
        }
        testing_utils.eval_model(opt, skip_test=True)

    def test_exp_attn_train(self):
        opt = {
            'model': 'projects.light_whoami.agents.expanded_attention:ExpandedDecoderAttentionAndMultiObjectiveAgent',
            'mutators': 'clean_context_mutator,share_self_character',
            'expanded_attention_share_weights': True,
            'expanded_attention_type': 'automated_trainable_mask',
            'automated_expanded_attention_n_tokens': 10,
            'expanded_attention_classifier_model_file': TEST_RPA_RERANKER,
            'multiobjective_latent_representation': 'encoder_and_decoder',
            'n_multiobjective_heads': 4,
            'n_multiobjective_layers': 2,
            **COMMON_OPT,
            'task': 'projects.light_whoami.task.agents:MultiObjectiveTeacher',
        }
        opt.pop('num_examples')
        opt.update(TRAIN_COMMON_OPT)
        testing_utils.train_model(opt)
