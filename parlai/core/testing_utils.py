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


DEBUG = False  # change this to true to print to stdout anyway


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


@contextlib.contextmanager
def capture_output():
    if DEBUG:
        yield
    else:
        sio = io.StringIO()
        with contextlib.redirect_stdout(sio), contextlib.redirect_stderr(sio):
            yield sio


@contextlib.contextmanager
def tempdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def train_model(opt):
    """
    Runs through a TrainLoop.

    :return: (stdout, stderr, valid_results, test_results)
    :rtype: (str, str, dict, dict)

    If model_file is not in opt, then this helper will create a temporary
    to store the model, dict, etc.
    """
    import parlai.scripts.train_model as tms

    with capture_output() as output:
        with tempdir() as tmpdir:
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
        output.getvalue(),
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

    if popt.get('model_file') and not popt.get('dict_file'):
        popt['dict_file'] = popt['model_file'] + '.dict'

    with capture_output() as output:
        popt['datatype'] = 'valid'
        valid = ems.eval_model(popt)
        popt['datatype'] = 'test'
        test = ems.eval_model(popt)

    return (
        output.getvalue(),
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
    with capture_output() as _:
        download_models(opt, model_filenames, 'unittest')
