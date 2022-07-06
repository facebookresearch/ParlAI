#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.opt import Opt
from parlai.tasks.integration_tests.agents import NUM_TEST, EXAMPLE_SIZE
from parlai.utils.conversations import Conversations
import parlai.utils.testing as testing_utils

import os
from typing import Dict, Any
import unittest

_TASK = 'integration_tests:variable_length'

# we don't need a real agent, since we're only checking the number examples
# is correct
_DEFAULT_OPTIONS = {
    'dict_file': 'zoo:unittest/transformer_generator2/model.dict',
    'dict_tokenizer': 'space',
    'batchsize': 64,
    'dynamic_batching': 'full',
    'num_epochs': 0.1,
    'truncate': 8,
    'model': 'parlai.agents.test_agents.test_agents:SilentTorchAgent',
    'task': _TASK,
}

_RANKER_OPTIONS = {
    'dict_file': 'zoo:unittest/transformer_generator2/model.dict',
    'dict_tokenizer': 'space',
    'batchsize': 32,
    'num_epochs': 0.1,
    'n_layers': 1,
    'n_heads': 1,
    'candidates': 'batch',
    'ffn_size': 4,
    'embedding_size': 4,
    'task': _TASK,
    'truncate': 8,
    'model': 'transformer/ranker',
}


# TODO tests to write:
# - multiple validation runs, streaming/not streaming
# - ranking model


class TestDynamicBatching(unittest.TestCase):
    def _test_correct_processed(self, num_goal: int, **kwargs: Dict[str, Any]):
        opt = Opt({**_DEFAULT_OPTIONS, **kwargs})
        valid_report, test_report = testing_utils.train_model(opt)
        self.assertEqual(valid_report['exs'], num_goal)
        self.assertEqual(test_report['exs'], num_goal)

    def test_no_truncate(self):
        with self.assertRaises(ValueError):
            testing_utils.train_model(Opt({**_DEFAULT_OPTIONS, **{'truncate': -1}}))

    def test_no_batch_act(self):
        """
        Fail when the agent doesn't support dynamic batching.
        """
        with self.assertRaises(TypeError):
            testing_utils.train_model(model='repeat_label', task=_TASK)

        with self.assertRaises(TypeError):
            testing_utils.eval_model(model='repeat_label', task=_TASK)

    def test_ranking(self):
        testing_utils.train_model(
            Opt(datatype='train', dynamic_batching='full', **_RANKER_OPTIONS)
        )

    def test_ranking_streaming(self):
        testing_utils.train_model(
            Opt(datatype='train:stream', dynamic_batching='full', **_RANKER_OPTIONS)
        )

    def test_training(self):
        self._test_correct_processed(NUM_TEST, datatype='train')

    def test_streaming(self):
        self._test_correct_processed(NUM_TEST, datatype='train:stream')

    def test_multiworld(self):
        self._test_correct_processed(
            NUM_TEST + NUM_TEST * EXAMPLE_SIZE,
            task='integration_tests:variable_length,integration_tests:multiturn',
        )

    def test_multiworld_stream(self):
        self._test_correct_processed(
            NUM_TEST + NUM_TEST * EXAMPLE_SIZE,
            task='integration_tests:variable_length,integration_tests:multiturn',
            datatype='train:stream',
        )

    def test_world_logging(self):
        with testing_utils.tempdir() as tmpdir:
            save_report = os.path.join(tmpdir, 'report')
            testing_utils.eval_model(
                dict(
                    model_file='zoo:unittest/transformer_generator2/model',
                    task='integration_tests:multiturn_candidate',
                    world_logs=save_report,
                    report_filename=save_report,
                    truncate=1024,
                    dynamic_batching='full',
                    batchsize=4,
                )
            )
            convo_fle = str(save_report) + '.jsonl'
            convos = Conversations(convo_fle)
            for convo in convos:
                self.assertEquals(len(convo), 2 * 4)  # each episode is 4 turns
                # now assert that they are all from the same dynamic batch index
                dyn_batch_idx = convo[0]['dyn_batch_idx']
                for i, turn in enumerate(convo):
                    if i % 2 == 0 and i > 0:
                        # we log the batch index in the teacher acts only
                        self.assertEquals(dyn_batch_idx, turn['dyn_batch_idx'])

    def test_world_logging_buffersize(self):
        """
        Test world logging with dynamic batching.

        Checks when the number of examples exceeds the buffersize.
        """
        with testing_utils.tempdir() as tmpdir:
            save_report = os.path.join(tmpdir, 'report')
            testing_utils.eval_model(
                dict(
                    model_file='zoo:unittest/transformer_generator2/model',
                    task='integration_tests:RepeatTeacher:2000',
                    world_logs=save_report + '.jsonl',
                    report_filename=save_report,
                    truncate=1024,
                    dynamic_batching='full',
                    batchsize=4,
                ),
                valid_datatype='train:evalmode',
                skip_test=True,
            )
            convo_fle = str(save_report) + '.jsonl'
            convos = Conversations(convo_fle)
            # we expect there to be 2000 episodes logged in the convos
            self.assertEquals(len(convos), 2000)

    def test_weird_batchsize(self):
        # intentionally a difficult number
        self._test_correct_processed(NUM_TEST, batchsize=7)

    def test_batchsize4(self):
        # intentionally an edgecase in the world
        self._test_correct_processed(NUM_TEST, batchsize=4)

    def test_chunky(self):
        """
        Test dynamic batching with chunk teachers end to end.
        """
        self._test_correct_processed(
            NUM_TEST,
            model='test_agents/unigram',  # important we use a real model here
            task='integration_tests:chunky',
            datatype='train:stream',
            num_epochs=2,  # important we use num epochs > 1
        )


class TestBatchSort(unittest.TestCase):
    def _test_correct_processed(self, num_goal: int, **kwargs: Dict[str, Any]):
        opt = Opt({**_DEFAULT_OPTIONS, **kwargs})
        opt['dynamic_batching'] = 'batchsort'
        valid_report, test_report = testing_utils.train_model(opt)
        self.assertEqual(valid_report['exs'], num_goal)
        self.assertEqual(test_report['exs'], num_goal)

    def test_no_batch_act(self):
        """
        Fail when the agent doesn't support dynamic batching.
        """
        with self.assertRaises(TypeError):
            testing_utils.train_model(model='repeat_label', task=_TASK)

        with self.assertRaises(TypeError):
            testing_utils.eval_model(model='repeat_label', task=_TASK)

    def test_ranking(self):
        testing_utils.train_model(
            Opt(datatype='train', dynamic_batching='batchsort', **_RANKER_OPTIONS)
        )

    def test_ranking_streaming(self):
        testing_utils.train_model(
            Opt(
                datatype='train:stream', dynamic_batching='batchsort', **_RANKER_OPTIONS
            )
        )

    def test_training(self):
        self._test_correct_processed(NUM_TEST, datatype='train')

    def test_streaming(self):
        self._test_correct_processed(NUM_TEST, datatype='train:stream')

    def test_multiworld(self):
        self._test_correct_processed(
            NUM_TEST + NUM_TEST * EXAMPLE_SIZE,
            task='integration_tests:variable_length,integration_tests:multiturn',
        )

    def test_multiworld_stream(self):
        self._test_correct_processed(
            NUM_TEST + NUM_TEST * EXAMPLE_SIZE,
            task='integration_tests:variable_length,integration_tests:multiturn',
            datatype='train:stream',
        )

    def test_weird_batchsize(self):
        # intentionally a difficult number
        self._test_correct_processed(NUM_TEST, batchsize=7)

    def test_batchsize4(self):
        # intentionally an edgecase in the world
        self._test_correct_processed(NUM_TEST, batchsize=4)


class TestTinyDataset(unittest.TestCase):
    """
    Test Dyanmic batching when we have a world size fewer than the number of available
    GPUs.
    """

    @testing_utils.skipUnlessGPU
    def test_tiny_model(self):
        import parlai.scripts.multiprocessing_train as mp_train

        from torch.cuda import device_count

        if device_count() < 2:
            raise unittest.SkipTest("Need at least 2 GPUs to test")

        valid_report, test_report = mp_train.MultiProcessTrain.main(
            model='test_agents/unigram',
            dict_file='zoo:unittest/transformer_generator2/model.dict',
            dict_tokenizer='space',
            task='integration_tests:tiny',
            batchsize=2,
            dynamic_batching='full',
            truncate=8,
            verbose=True,
            validation_every_n_steps=5,
            max_train_steps=10,
        )

        assert valid_report['exs'] == 1
        assert test_report['exs'] == 1


if __name__ == '__main__':
    unittest.main()
