#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
General utilities for helping writing ParlAI unit and integration tests.
"""

import os
import unittest
import contextlib
import tempfile
import shutil
import io


try:
    import torch
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.device_count() > 0
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False


def skipUnlessTorch(testfn, reason='pytorch is not installed'):
    """Decorator for skipping a test if torch is not installed."""
    return unittest.skipUnless(TORCH_AVAILABLE, reason)(testfn)


def skipIfGPU(testfn, reason='Test is CPU-only'):
    """
    Decorator for skipping a test if a GPU is available.

    Useful for disabling hogwild tests.
    """
    return unittest.skipIf(GPU_AVAILABLE, reason)(testfn)


def skipUnlessGPU(testfn, reason='Test requires a GPU'):
    """Decorator for skipping a test if no GPU is available."""
    return unittest.skipIf(not GPU_AVAILABLE, reason)(testfn)


def skipIfTravis(testfn, reason='Test disabled in Travis'):
    """Decorator for skipping a test if running on Travis."""
    return unittest.skipIf(bool(os.environ('TRAVIS')), reason)


class TempDir(object):
    """
    Syntactic sugar allowing for "with TempDir() as dir:"
    """
    def __enter__(self):
        self.dir = tempfile.mkdtemp()
        return self.dir

    def __exit__(self, exception_type, exception_value, traceback):
        shutil.rmtree(self.dir)

    def __str__(self):
        return self.dir

    def __repr__(self):
        return self.dir


def train_model(opt):
    """
    Runs through a TrainLoop.

    :return: (stdout, stderr, valid_results, test_results)
    :rtype: (str, str, dict, dict)

    If model_file is not in opt, then this helper will create a temporary
    to store the model, dict, etc.
    """
    import parlai.scripts.train_model as tms

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        with TempDir() as tmpdir:
            if 'model_file' not in opt:
                opt['model_file'] = os.path.join(tmpdir, 'model')
            if 'dict_file' not in opt:
                opt['dict_file'] = os.path.join(tmpdir, 'model.dict')
            parser = tms.setup_args()
            parser.set_params(**opt)
            popt = parser.parse_args(print_args=False)
            tl = tms.TrainLoop(popt)
            valid, test = tl.train()

    return (
        stdout.getvalue(),
        stderr.getvalue(),
        valid,
        test,
    )


def eval_model(opt):
    """
    Runs through an evaluation loop.

    :return: (stdout, stderr, valid_results, test_results)
    :rtype: (str, str, dict, dict)

    If model_file is not in opt, then this helper will create a temporary directory
    to store the model files, and clean up afterwards. You can keep the directory
    by disabling autocleanup
    """

    import parlai.scripts.eval_model as ems
    parser = ems.setup_args()
    parser.set_params(**opt)
    popt = parser.parse_args(print_args=False)

    if 'dict_file' not in popt and 'model_file' in popt:
        popt['dict_file'] = popt['model_file'] + '.dict'

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        opt['datatype'] = 'valid'
        valid = ems.eval_model(popt)
        opt['datatype'] = 'test'
        test = ems.eval_model(popt)

    return (
        stdout.getvalue(),
        stderr.getvalue(),
        valid,
        test,
    )


def download_unittest_models():
    from parlai.core.params import ParlaiParser
    from parlai.core.build_data import download_models
    opt = ParlaiParser().parse_args(print_args=False)
    model_filenames = [
        'seq2seq.tar.gz',
        'transformer_ranker.tar.gz',
        'transformer_generator.tar.gz'
    ]
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        download_models(opt, model_filenames, 'unittest')
