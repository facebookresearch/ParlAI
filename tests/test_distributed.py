#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import unittest
import torch.distributed as dist
import parlai.utils.testing as testing_utils
import parlai.scripts.build_dict as build_dict
import parlai.scripts.multiprocessing_train as mp_train
import parlai.tasks.integration_tests.agents as inttests

BATCHSIZE = 4


def _forced_parse(parser, opt):
    parser.set_params(**opt)
    parser.set_params(log_every_n_sec=10)
    popt = parser.parse_args([])
    # in some rare cases, like for instance if the model class also
    # overrides its default params, the params override will not
    # be taken into account.
    for k, v in opt.items():
        popt[k] = v
    return popt


@testing_utils.skipUnlessGPU
class TestDistributed(unittest.TestCase):
    _base_config = dict(
        task='integration_tests:overfit',
        model='transformer/generator',
        optimizer='adam',
        validation_metric='ppl',
        skip_generation=True,
        learningrate=1e-2,
        batchsize=BATCHSIZE,
        validation_every_n_epochs=5,
        num_epochs=100,
        n_layers=1,
        n_heads=1,
        ffn_size=32,
        embedding_size=8,
        verbose=True,
    )

    def setUp(self):
        print(f'[Setting up test {self._testMethodName}]')

    def _distributed_train_model(self, opt):
        with testing_utils.tempdir() as tmpdir:
            if 'model_file' not in opt:
                opt['model_file'] = os.path.join(tmpdir, 'model')
            if 'dict_file' not in opt:
                opt['dict_file'] = os.path.join(tmpdir, 'model.dict')

            parser = mp_train.setup_args()
            popt = _forced_parse(parser, opt)

            # we need a prebuilt dictionary
            parser = build_dict.setup_args()
            build_dict.build_dict(popt)

            valid, test = mp_train.launch_and_train(popt, 31338)
            dist.destroy_process_group()

        return (valid, test)

    @testing_utils.retry()
    def test_generator_distributed(self):
        config = copy.deepcopy(self._base_config)
        valid, test = self._distributed_train_model(config)

        self.assertLessEqual(valid['ppl'], 1.50)
        self.assertLessEqual(test['ppl'], 1.50)

        # Tests that DialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        self.assertEqual(valid['exs'].value(), BATCHSIZE)
        self.assertEqual(test['exs'].value(), BATCHSIZE)

    @testing_utils.retry()
    def test_multitask_distributed(self):
        config = copy.deepcopy(self._base_config)
        config['num_epochs'] = 50
        config['task'] = 'integration_tests:overfit,integration_tests:overfit_multiturn'
        config['dynb'] = 'full'
        valid, test = self._distributed_train_model(config)

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
        config = copy.deepcopy(self._base_config)
        config['task'] = 'integration_tests'
        config['num_epochs'] = 0.01
        config['validation_max_exs'] = 90
        config['short_final_eval'] = True
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
        self.assertEqual(valid['exs'].value(), 96)
        self.assertEqual(test['exs'].value(), 96)

    def test_distributed_eval_stream_mode(self):
        config = copy.deepcopy(self._base_config)
        config['task'] = 'integration_tests'
        config['num_epochs'] = 0.01
        config['datatype'] = 'train:stream'
        valid, test = self._distributed_train_model(config)

        # Tests that StreamDialogData.get() is doing the right thing
        # Ensure no duplication of examples among workers
        # It would be 200 if each worker did all the examples
        self.assertEqual(valid['exs'].value(), inttests.NUM_TEST)
        self.assertEqual(test['exs'].value(), inttests.NUM_TEST)

    def test_distributed_eval_stream_mode_max_exs(self):
        config = copy.deepcopy(self._base_config)
        config['task'] = 'integration_tests'
        config['num_epochs'] = 0.01
        config['datatype'] = 'train:stream'
        config['validation_max_exs'] = 90
        config['short_final_eval'] = True

        valid, test = self._distributed_train_model(config)

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
        config = copy.deepcopy(self._base_config)
        config['task'] = 'integration_tests'
        config['num_epochs'] = 0.01
        config['datatype'] = 'train:stream'
        config['dynamic_batching'] = 'full'
        config['truncate'] = 16

        valid, test = self._distributed_train_model(config)
        assert valid['exs'].value() == inttests.NUM_TEST
        assert test['exs'].value() == inttests.NUM_TEST

    def test_chunked_teacher(self):
        config = copy.deepcopy(self._base_config)
        config['task'] = 'integration_tests'
        config['num_epochs'] = 0.01
        config['datatype'] = 'train:stream'
        config['num_epochs'] = 5
        config['dynamic_batching'] = None

        valid, test = self._distributed_train_model(config)
        assert valid['exs'].value() == inttests.NUM_TEST
        assert test['exs'].value() == inttests.NUM_TEST

    def test_no_model_parallel(self):
        """
        Checks that we throw an error when combining mp_train with.

        --model-parallel true.
        """
        config = copy.deepcopy(self._base_config)
        config['model_parallel'] = True
        for m in [
            'transformer/generator',
            'transformer/ranker',
            'transformer/classifier',
        ]:
            config['model'] = m
            try:
                _ = self._distributed_train_model(config)
            except RuntimeError:
                pass
            else:
                self.fail('Did not raise RuntimeError')
            finally:
                dist.destroy_process_group()


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
        dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
