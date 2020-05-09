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


def _forced_parse(parser, opt):
    parser.set_params(**opt)
    parser.set_params(log_every_n_sec=10)
    popt = parser.parse_args([], print_args=False)
    # in some rare cases, like for instance if the model class also
    # overrides its default params, the params override will not
    # be taken into account.
    for k, v in opt.items():
        popt[k] = v
    return popt


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

    def tearDown(self):
        # we need to de-initialize the distributed world, otherwise other
        # tests will they're we're distributed when we're really not.
        dist.destroy_process_group()

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

            valid, test = mp_train.launch_and_train(popt, 31337)

        return (valid, test)

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
        config = copy.deepcopy(self._base_config)
        config['task'] = 'integration_tests:nocandidate,integration_tests:multiturn'
        config['dynb'] = 'full'
        config['skip_generation'] = 'true'
        valid, test = self._distributed_train_model(config)

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
        config = copy.deepcopy(self._base_config)
        config['datatype'] = 'train:stream'
        valid, test = self._distributed_train_model(config)

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


if __name__ == '__main__':
    unittest.main()
