#!/usr/bin/env python

import unittest
import parlai.core.testing_utils as testing_utils


class TestCreateAgent(unittest.TestCase):
    def test_command(self):
        # just ensure no crash
        testing_utils.train_model(
            {
                'task': 'babi:task10k:1',
                'batchsize': 32,
                'validation_cutoff': 0.95,
                'no_cuda': True,
                'validation_every_n_secs': 30,
                'num_epochs': 2,
                'model': 'example_seq2seq',
            }
        )


if __name__ == '__main__':
    unittest.main()
