#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest


def _clear_cmdline_args(fn):
    """
    Helper decorator to make sure 'python setup.py test' doesn't look like
    a parlai call.
    """
    import sys
    sys.argv = sys.argv[:1]
    return fn


@_clear_cmdline_args
def datatests():
    """
    Tests for data integrity. Runs on CircleCI.

    Separate to help distinguish failure reasons.
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests/datatests')
    return test_suite


@_clear_cmdline_args
def nightly_gpu():
    """Nightly GPU tests. Runs on internal infra."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests/nightly/gpu')
    return test_suite


@_clear_cmdline_args
def unittests():
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


if __name__ == '__main__':
    unittest.run(unittests())
