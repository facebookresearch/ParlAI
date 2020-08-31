#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import torch.distributed as dist
from parlai.core.opt import Opt
import parlai.utils.testing as testing_utils
import parlai.scripts.build_dict as build_dict
import parlai.scripts.multiprocessing_train as mp_train


@testing_utils.skipUnlessGPU
class TestDistributed(unittest.TestCase):
    _base_config = Opt(
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
        verbose=True,
    )

    def setUp(self):
        print(f'[Setting up test {self._testMethodName}]')

    def tearDown(self):
        # we need to de-initialize the distributed world, otherwise other
        # tests will they're we're distributed when we're really not.
        dist.destroy_process_group()

    def _distributed_train_model(self, opt):
        with testing_utils.tempdir() as tmpdir:
            if 'model_file' not in opt:
                opt = opt.fork(model_file=os.path.join(tmpdir, 'model'))
            if 'dict_file' not in opt:
                opt = opt.fork(dict_file=opt['model_file'] + '.dict')

            # we need a prebuilt dictionary
            build_dict.BuildDict.main(
                task=opt['task'],
                dict_file=opt['dict_file'],
                model_file=opt['model_file'],
            )
            valid, test = mp_train.MultiProcessTrain.main(**opt)

        return valid, test

    def test_generator_distributed(self):
        valid, test = self._distributed_train_model(self._base_config)

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
        config = self._base_config.fork(
            task='integration_tests:nocandidate,integration_tests:multiturn',
            skip_generation='true',
        )
        valid, test = self._distributed_train_model(config)

        self.assertLessEqual(valid['ppl'], 1.20)
        self.assertLessEqual(test['ppl'], 1.20)

        # Tests that DialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        self.assertEqual(valid['exs'].value(), 500)
        self.assertEqual(test['exs'].value(), 500)

    def test_distributed_eval_max_exs(self):
        config = self._base_config.fork(validation_max_exs=90, short_final_eval=True)
        valid, test = self._distributed_train_model(config)

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
        config = self._base_config.fork(datatype='train:stream')
        valid, test = self._distributed_train_model(config)

        # Tests that StreamDialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        self.assertEqual(valid['exs'].value(), 100)
        self.assertEqual(test['exs'].value(), 100)

    def test_distributed_eval_stream_mode_max_exs(self):
        config = self._base_config.fork(
            datatype='train:stream', validation_max_exs=90, short_final_eval=True
        )

        valid, test = self._distributed_train_model(config)

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

    def test_chunked_dynamic_teacher(self):
        config = self._base_config.fork(
            datatype='train:stream', dynamic_batching='full', truncate=16,
        )

        valid, test = self._distributed_train_model(config)
        assert valid['exs'].value() == 100
        assert test['exs'].value() == 100

    def test_chunked_teacher(self):
        config = self._base_config.fork(
            datatype='train:stream', num_epochs=5, dynamic_batching=None,
        )

        valid, test = self._distributed_train_model(config)
        assert valid['exs'].value() == 100
        assert test['exs'].value() == 100

    def test_no_model_parallel(self):
        """
        Checks that we throw an error when combining mp_train with.

        --model-parallel true.
        """
        config = self._base_config.fork(model_parallel=True)
        for m in [
            'transformer/generator',
            'transformer/ranker',
            'transformer/classifier',
        ]:
            with self.assertRaises(RuntimeError):
                self._distributed_train_model(config.fork(model=m))


if __name__ == '__main__':
    unittest.main()
