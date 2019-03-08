#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.core.testing_utils as testing_utils


def _clear_cmdline_args(fn):
    """
    Helper decorator to make sure 'python setup.py test' doesn't look like
    a parlai call.
    """
    import sys
    sys.argv = sys.argv[:1]
    return fn


@_clear_cmdline_args
def nightly_cpu():
    """Nightly CPU tests. Runs on CircleCI"""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests/nightly/cpu')
    return test_suite


@_clear_cmdline_args
def nightly_gpu():
    """Nightly GPU tests. Runs on internal infra."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests/nightly/gpu')
    return test_suite


@_clear_cmdline_args
def short():
    """Short tests, found in tests root directory."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    return test_suite


@_clear_cmdline_args
def mturk():
    """Mechanical Turk tests."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("parlai/mturk/core/test/")
    return test_suite


@_clear_cmdline_args
def oncommit():
    """Tests needed to pass Continuous Integration."""
    test_suite = unittest.TestSuite()
    test_suite.addTests(short())
    changed_files = testing_utils.git_changed_files(skip_nonexisting=False)
    if any('parlai/mturk' in fn for fn in changed_files):
        # if any mturk stuff changed, run those tests too
        test_suite.addTests(mturk())
    return test_suite


if __name__ == '__main__':
    unittest.run(oncommit())
