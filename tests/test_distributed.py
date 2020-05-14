#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
import parlai.utils.testing as testing_utils


@testing_utils.skipUnlessGPU
class TestDistributed(unittest.TestCase):
    _base_config = dict(
        task='integration_tests:nocandidate',
        model='transformer/generator',
        optimizer='adamax',
        validation_metric='ppl',
        learningrate=7e-3,
        batchsize=7,
        validation_every_n_epochs=5,
        num_epochs=20,
        n_layers=1,
        n_heads=1,
        ffn_size=32,
        embedding_size=32,
        beam_size=1,
        verbose=True,
    )

    def setUp(self):
        print(f'[Setting up test {self._testMethodName}]')

    def test_generator_distributed(self):
        valid, test = testing_utils.distributed_train_model(self._base_config)

        self.assertLessEqual(valid['ppl'], 1.20)
        self.assertGreaterEqual(valid['bleu-4'], 0.95)
        self.assertLessEqual(test['ppl'], 1.20)
        self.assertGreaterEqual(test['bleu-4'], 0.95)

        # Tests that DialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        self.assertEqual(valid['exs'].value(), 100)
        self.assertEqual(test['exs'].value(), 100)

    def test_multitask_distributed(self):
        config = copy.deepcopy(self._base_config)
        config['task'] = 'integration_tests:nocandidate,integration_tests:multiturn'
        config['dynb'] = 'full'
        config['skip_generation'] = 'true'
        valid, test = testing_utils.distributed_train_model(config)

        self.assertLessEqual(valid['ppl'], 1.20)
        self.assertLessEqual(test['ppl'], 1.20)

        # Tests that DialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        self.assertEqual(valid['exs'].value(), 500)
        self.assertEqual(test['exs'].value(), 500)

    def test_distributed_eval_max_exs(self):
        config = copy.deepcopy(self._base_config)
        config['validation_max_exs'] = 90
        config['short_final_eval'] = True
        valid, test = testing_utils.distributed_train_model(config)

        # Tests that DialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        # Note: we decided that it was OK for the total count to be slightly off
        # when using validation_max_exs and distributed.
        # It's > 90 b/c there are two workers, told to do 45 each, & BatchWorld
        # parley() does batchsize examples each time, so each worker will do 49
        # example. In the future, if we fix VME, this assert should be changed
        # to exactly 90.
        self.assertEqual(valid['exs'].value(), 98)
        self.assertEqual(test['exs'].value(), 98)

    def test_distributed_eval_stream_mode(self):
        config = copy.deepcopy(self._base_config)
        config['datatype'] = 'train:stream'
        valid, test = testing_utils.distributed_train_model(config)

        # Tests that StreamDialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        self.assertEqual(valid['exs'].value(), 100)
        self.assertEqual(test['exs'].value(), 100)

    def test_distributed_eval_stream_mode_max_exs(self):
        config = copy.deepcopy(self._base_config)
        config['datatype'] = 'train:stream'
        config['validation_max_exs'] = 90
        config['short_final_eval'] = True

        valid, test = testing_utils.distributed_train_model(config)

        # Tests that StreamDialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        # As in the test above:
        # It does 98 instead of 90 b/c there are two workers, told to do 45
        # each, and BatchWorld parley() does batchsize examples each time, so
        # each worker will do 49 examples.
        # In the future, if we fix VME, this assert should be changed to
        # exactly 90.
        self.assertEqual(valid['exs'].value(), 98)
        self.assertEqual(test['exs'].value(), 98)


if __name__ == '__main__':
    unittest.main()
