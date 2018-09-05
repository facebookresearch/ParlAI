# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import unittest
import io
import contextlib
import tempfile
import os
import shutil

from parlai.scripts.train_model import TrainLoop, setup_args


class TestFairseq(unittest.TestCase):
    """Checks that fairseq can learn some very basic tasks."""

    def test_labelcands(self):
        outdir = tempfile.mkdtemp()
        parser = setup_args()
        parser.set_defaults(
            model_file=os.path.join(outdir, "model"),
            task='unittest:CandidateTeacher',
            model='fairseq',
            arch='lstm_wiseman_iwslt_de_en',
            lr=3e-4,
            batchsize=16,
            num_epochs=10,
            rank_candidates=True,
            skip_generation=True,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            tl = TrainLoop(parser)
            valid, test = tl.train()

            shutil.rmtree(outdir)

            assert valid['hits@1'] > 0.95, "valid hits@1 = %f" % (valid['hits@1'])
            assert test['hits@1'] > 0.95, "test hits@1 = %f" % (test['hits@1'])

    def test_generation(self):
        outdir = tempfile.mkdtemp()
        parser = setup_args()
        parser.set_defaults(
            model_file=os.path.join(outdir, "model"),
            task='unittest:NocandidateTeacher',
            model='fairseq',
            arch='lstm_wiseman_iwslt_de_en',
            lr=3e-4,
            batchsize=16,
            num_epochs=10,
            rank_candidates=False,
            skip_generation=False,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            tl = TrainLoop(parser)
            valid, test = tl.train()
            shutil.rmtree(outdir)
            assert valid['ppl'] < 1.2, "valid ppl = %f" % (valid['ppl'])
            assert test['ppl'] < 1.2, "test ppl = %f" % (test['ppl'])


if __name__ == '__main__':
    unittest.main()
