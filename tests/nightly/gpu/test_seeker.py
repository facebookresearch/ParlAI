#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import parlai.scripts.eval_model as ems
import parlai.utils.testing as testing_utils

from projects.seeker.utils import (
    R2C2_BASE_400M,
    R2C2_BLENDERBOT_400M,
    SEEKER_DIALOGUE_400M,
    SEEKER_LM_MED,
)

LOCAL_TEST = False

search_task = 'projects.seeker.tasks.search_query'
dialogue_task = 'projects.seeker.tasks.dialogue:WoiDialogueTeacher'
knowledge_task_search = 'projects.seeker.tasks.knowledge:WoiKnowledgeTeacher'
knowledge_task_no_search = 'projects.seeker.tasks.knowledge:Convai2KnowledgeTeacher'
all_tasks = ','.join(
    [search_task, dialogue_task, knowledge_task_no_search, knowledge_task_search]
)


TRAIN_COMMON_OPT = {
    'max_train_steps': 16,
    'validation_max_exs': 5,
    'short_final_eval': True,
    'truncate': 16,
    'text_truncate': 16,
    'label_truncate': 16,
    'model': 'transformer/generator',
    'init_model': 'zoo:unittest/transformer_generator2/model',
    'dict_file': 'zoo:unittest/transformer_generator2/model.dict',
    'n_layers': 2,
    'n_heads': 2,
    'embedding_size': 32,
    'ffn_size': 128,
    'n_positions': 1024,
    'dict_tokenizer': 're',
    'optimizer': 'sgd',
    'skip_generation': True,
}


class TestTrain(unittest.TestCase):
    """
    Test training.
    """

    def test_training(self):
        opt = {
            'model': 'projects.seeker.agents.seeker:ComboFidGoldDocumentAgent',
            'task': all_tasks,
            **TRAIN_COMMON_OPT,
        }
        testing_utils.train_model(opt)


class TestDialogueZoo(unittest.TestCase):
    def test_seeker_dialogue(self):
        opt = {
            'model_file': SEEKER_DIALOGUE_400M,
            'task': 'integration_tests:nocandidate',
            'init_opt': 'gen/seeker_dialogue',
            'search_decision': 'never',
            'num_examples': 2,
            'datatype': 'valid',
            'search_server': 'blah',
        }
        ems.EvalModel.main(**opt)


class TestLMZoo(unittest.TestCase):
    def test_seeker_lm(self):
        opt = {
            'model_file': SEEKER_LM_MED,
            'task': 'integration_tests:nocandidate',
            'init_opt': 'gen/seeker_lm',
            'search_decision': 'never',
            'num_examples': 2,
            'datatype': 'valid',
            'search_server': 'blah',
        }
        ems.EvalModel.main(**opt)


class TestR2C2Zoo(unittest.TestCase):
    def test_base(self):
        opt = {
            'model_file': R2C2_BASE_400M,
            'task': 'integration_tests:nocandidate',
            'init_opt': 'gen/blenderbot',
            'num_examples': 2,
            'datatype': 'valid',
        }
        ems.EvalModel.main(**opt)

    def test_blenderbot(self):
        opt = {
            'model_file': R2C2_BLENDERBOT_400M,
            'task': 'integration_tests:nocandidate',
            'init_opt': 'gen/blenderbot',
            'num_examples': 2,
            'datatype': 'valid',
        }
        ems.EvalModel.main(**opt)
