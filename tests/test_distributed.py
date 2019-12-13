#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import torch.distributed as dist
import parlai.utils.testing as testing_utils
import parlai.scripts.build_dict as build_dict
import parlai.scripts.multiprocessing_train as mp_train


def _forced_parse(parser, opt):
    parser.set_params(**opt)
    parser.set_params(log_every_n_sec=10)
    popt = parser.parse_args(print_args=False)
    # in some rare cases, like for instance if the model class also
    # overrides its default params, the params override will not
    # be taken into account.
    for k, v in opt.items():
        popt[k] = v
    return popt


@testing_utils.skipUnlessGPU
class TestDistributed(unittest.TestCase):
    def _distributed_train_model(self, opt):
        with testing_utils.capture_output() as output:
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

        return (output.getvalue(), valid, test)

    def tearDown(self):
        # we need to de-initialize the distributed world, otherwise other
        # tests will they're we're distributed when we're really not.
        dist.destroy_process_group()

    def test_generator_distributed(self):
        stdout, valid, test = self._distributed_train_model(
            dict(
                task='integration_tests:nocandidate',
                model='transformer/generator',
                optimizer='adamax',
                learningrate=7e-3,
                batchsize=32,
                validation_every_n_epochs=5,
                num_epochs=20,
                n_layers=1,
                n_heads=1,
                ffn_size=32,
                embedding_size=32,
                beam_size=1,
            )
        )

        self.assertLessEqual(
            valid['ppl'], 1.20, "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertGreaterEqual(
            valid['bleu-4'],
            0.95,
            "valid blue = {}\nLOG:\n{}".format(valid['bleu-4'], stdout),
        )
        self.assertLessEqual(
            test['ppl'], 1.20, "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )
        self.assertGreaterEqual(
            test['bleu-4'],
            0.95,
            "test bleu = {}\nLOG:\n{}".format(test['bleu-4'], stdout),
        )


if __name__ == '__main__':
    unittest.main()
