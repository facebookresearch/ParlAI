#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import parlai.utils.testing as testing_utils
import parlai.scripts.build_dict as build_dict
import parlai.scripts.multiprocessing_train as mp_train
import parlai.tasks.integration_tests.agents as inttests

BATCHSIZE = 4


class _AbstractTest(unittest.TestCase):
    def _distributed_train_model(self, **overrides):
        opt = {**self.base_config, **overrides}
        with testing_utils.tempdir() as tmpdir:
            if 'model_file' not in opt:
                opt['model_file'] = os.path.join(tmpdir, 'model')
            if 'dict_file' not in opt:
                opt['dict_file'] = os.path.join(tmpdir, 'model.dict')

            parser = mp_train.setup_args()
            popt = parser.parse_kwargs(**opt)

            # we need a prebuilt dictionary
            parser = build_dict.setup_args()
            build_dict.build_dict(popt)

            valid, test = mp_train.launch_and_train(popt)

        return (valid, test)


@testing_utils.skipUnlessGPU
class TestDistributed(_AbstractTest):
    base_config = dict(
        task='integration_tests:overfit',
        model='transformer/generator',
        optimizer='adam',
        validation_metric='ppl',
        skip_generation=True,
        learningrate=1e-2,
        batchsize=BATCHSIZE,
        validation_every_n_epochs=5,
        num_epochs=150,
        n_layers=1,
        n_heads=1,
        ffn_size=32,
        embedding_size=8,
        verbose=True,
    )

    def test_generator_distributed(self):
        valid, test = self._distributed_train_model()

        self.assertLessEqual(valid['ppl'], 1.60)
        self.assertLessEqual(test['ppl'], 1.60)

        # Tests that DialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        self.assertEqual(valid['exs'].value(), BATCHSIZE)
        self.assertEqual(test['exs'].value(), BATCHSIZE)

    def test_multitask_distributed(self):
        valid, test = self._distributed_train_model(
            num_epochs=50,
            task='integration_tests:overfit,integration_tests:overfit_multiturn',
            truncate=16,
        )

        self.assertLessEqual(valid['ppl'], 1.20)
        self.assertLessEqual(test['ppl'], 1.20)

        # Tests that DialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        self.assertEqual(
            valid['exs'].value(), BATCHSIZE * inttests.EXAMPLE_SIZE + BATCHSIZE
        )
        self.assertEqual(
            test['exs'].value(), BATCHSIZE * inttests.EXAMPLE_SIZE + BATCHSIZE
        )

    def test_distributed_eval_max_exs(self):
        valid, test = self._distributed_train_model(
            task='integration_tests',
            num_epochs=0.01,
            validation_max_exs=90,
            short_final_eval=True,
        )

        # Tests that DialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        # Note: we decided that it was OK for the total count to be slightly off
        # when using validation_max_exs and distributed.
        # It's > 90 b/c there are two workers, told to do 45 each, & BatchWorld
        # parley() does batchsize examples each time, so each worker will do 49
        # example. In the future, if we fix VME, this assert should be changed
        # to exactly 90.
        self.assertEqual(valid['exs'].value(), 96)
        self.assertEqual(test['exs'].value(), 96)

    def test_distributed_eval_stream_mode(self):
        valid, test = self._distributed_train_model(
            task='integration_tests', num_epochs=0.01, datatype='train:stream'
        )

        # Tests that StreamDialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        self.assertEqual(valid['exs'].value(), inttests.NUM_TEST)
        self.assertEqual(test['exs'].value(), inttests.NUM_TEST)

    def test_distributed_eval_stream_mode_max_exs(self):
        valid, test = self._distributed_train_model(
            task='integration_tests',
            num_epochs=0.01,
            datatype='train:stream',
            validation_max_exs=90,
            short_final_eval=True,
        )

        # Tests that StreamDialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        # As in the test above:
        # It does 96 instead of 90 b/c there are two workers, told to do 45
        # each, and BatchWorld parley() does batchsize examples each time, so
        # each worker will do 49 examples.
        # In the future, if we fix VME, this assert should be changed to
        # exactly 90.
        self.assertEqual(valid['exs'].value(), 96)
        self.assertEqual(test['exs'].value(), 96)

    def test_chunked_dynamic_teacher(self):
        valid, test = self._distributed_train_model(
            task='integration_tests',
            num_epochs=0.01,
            datatype='train:stream',
            dynamic_batching='full',
            truncate=16,
        )
        assert valid['exs'].value() == inttests.NUM_TEST
        assert test['exs'].value() == inttests.NUM_TEST

    def test_chunked_teacher(self):
        valid, test = self._distributed_train_model(
            task='integration_tests',
            datatype='train:stream',
            num_epochs=5,
            dynamic_batching=None,
        )
        assert valid['exs'].value() == inttests.NUM_TEST
        assert test['exs'].value() == inttests.NUM_TEST


@testing_utils.skipIfCircleCI
@testing_utils.skipUnlessGPU
class TestZero2(TestDistributed):
    """
    Integration tests for zero2 FSDP.
    """

    base_config = {**TestDistributed.base_config, 'ddp_backend': 'zero2'}


@unittest.skip
@testing_utils.skipUnlessGPU
class TestZero3(TestDistributed):
    # Not supported at this time. See:
    # https://github.com/facebookresearch/ParlAI/pull/3740
    base_config = {**TestDistributed.base_config, 'ddp_backend': 'zero3'}


@testing_utils.skipUnlessGPU
class TestNoModelParallel(_AbstractTest):
    base_config = dict(
        task='integration_tests:overfit',
        optimizer='sgd',
        validation_metric='loss',
        learningrate=1e-2,
        batchsize=BATCHSIZE,
        validation_every_n_epochs=1,
        num_epochs=1,
        n_layers=1,
        n_heads=1,
        ffn_size=32,
        embedding_size=8,
        verbose=True,
    )

    def test_no_model_parallel(self):
        """
        Checks that we throw an error when combining mp_train with --model-parallel.
        """
        for m in ['transformer/generator', 'transformer/ranker']:
            try:
                _ = self._distributed_train_model(model=m, model_parallel=True)
            except RuntimeError:
                pass
            else:
                self.fail('Did not raise RuntimeError')


@testing_utils.skipUnlessGPU
class TestDistributedEval(unittest.TestCase):
    def test_mp_eval(self):
        args = dict(
            task='integration_tests:multiturn_nocandidate',
            model='seq2seq',
            model_file='zoo:unittest/seq2seq/model',
            dict_file='zoo:unittest/seq2seq/model.dict',
            skip_generation=False,
            batchsize=8,
        )
        valid, _ = testing_utils.eval_model(args, skip_test=True)

        from parlai.scripts.multiprocessing_eval import MultiProcessEval

        valid_mp = MultiProcessEval.main(**args)

        for key in ['exs', 'ppl', 'token_acc', 'f1', 'bleu-4', 'accuracy']:
            self.assertAlmostEquals(
                valid[key].value(), valid_mp[key].value(), delta=0.001
            )


if __name__ == '__main__':
    unittest.main()
