#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


WOW_OPTIONS = {
    'task': 'wizard_of_wikipedia:Generator',
    'prepend_gold_knowledge': True,
    'model': 'image_seq2seq',
    'model_file': 'zoo:dodecadialogue/wizard_of_wikipedia_ft/model',
    'datatype': 'valid',
    'batchsize': 32,
    'inference': 'beam',
    'beam_block_ngram': 3,
    'beam_size': 10,
    'beam_min_length': 10,
    'skip_generation': False
}

CONVAI2_OPTIONS = {
    'task': 'convai2',
    'model': 'image_seq2seq',
    'model_file': 'zoo:dodecadialogue/convai2_ft/model',
    'datatype': 'valid',
    'batchsize': 32,
    'inference': 'beam',
    'beam_block_ngram': 3,
    'beam_size': 3,
    'beam_min_length': 10,
    'skip_generation': False
}

ED_OPTIONS = {
    'task': 'empathetic_dialogues',
    'model': 'image_seq2seq',
    'model_file': 'zoo:dodecadialogue/empathetic_dialogues_ft/model',
    'datatype': 'valid',
    'batchsize': 32,
    'inference': 'beam',
    'beam_block_ngram': 3,
    'beam_size': 5,
    'beam_min_length': 10,
    'skip_generation': False
}


@testing_utils.skipUnlessGPU
class TestDodecaModel(unittest.TestCase):
    """
    Checks that a few pre-trained Dodeca models give the correct results.
    """

    def test_wizard(self):
        """
        Test wiz of wikipedia.
        """
        valid, _ = testing_utils.eval_model(WOW_OPTIONS, skip_test=True)
        self.assertAlmostEqual(valid['ppl'], 8.5, places=1)
        self.assertAlmostEqual(valid['f1'], 0.379, places=3)

    def test_convai2(self):
        """
        Test ConvAI2.
        """
        valid, _ = testing_utils.eval_model(CONVAI2_OPTIONS, skip_test=True)
        self.assertAlmostEqual(valid['ppl'], 11.2, places=2)
        self.assertAlmostEqual(valid['f1'], 0.211, places=3)

    def test_ed(self):
        """
        Test empathetic_dialogues.
        """
        valid, _ = testing_utils.eval_model(ED_OPTIONS, skip_test=True)
        self.assertAlmostEqual(valid['ppl'], 11.1, places=2)
        self.assertAlmostEqual(valid['f1'], 0.197, places=3)

