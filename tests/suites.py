#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Various test loaders."""

import os
import unittest
import random
from itertools import chain


def _circleci_parallelism(suite):
    """Allow for parallelism in CircleCI for speedier tests.."""
    if int(os.environ.get('CIRCLE_NODE_TOTAL', 0)) <= 1:
        # either not running on circleci, or we're not using parallelism.
        return suite
    # tests are automatically sorted by discover, so we will get the same ordering
    # on all hosts.
    total = int(os.environ['CIRCLE_NODE_TOTAL'])
    index = int(os.environ['CIRCLE_NODE_INDEX'])

    # right now each test is corresponds to a /file/. Certain files are slower than
    # others, so we want to flatten it
    tests = [testfile._tests for testfile in suite._tests]
    tests = list(chain.from_iterable(tests))
    random.Random(42).shuffle(tests)
    tests = [t for i, t in enumerate(tests) if i % total == index]
    return unittest.TestSuite(tests)


def _clear_cmdline_args(fn):
    """Decorate to make sure 'python setup.py test' doesn't look like a parlai call."""
    import sys

    sys.argv = sys.argv[:1]
    return fn


@_clear_cmdline_args
def datatests():
    """
    Test for data integrity.

    Runs on CircleCI. Separate to help distinguish failure reasons.
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests/datatests')
    return test_suite


@_clear_cmdline_args
def nightly_gpu():
    """
    Nightly GPU tests.

    Runs on CircleCI nightly, and when [gpu] or [long] appears in a commit string.
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests/nightly/gpu')
    test_suite = _circleci_parallelism(test_suite)
    return test_suite


@_clear_cmdline_args
def unittests():
    """
    Short tests.

    Runs on CircleCI on every commit. Returns everything in the tests root directory.
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    test_suite = _circleci_parallelism(test_suite)
    return test_suite


@_clear_cmdline_args
def mturk():
    """Mechanical Turk tests."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("parlai/mturk/core/test/")
    return test_suite


@_clear_cmdline_args
def internal_tests():
    """Internal Tests"""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("parlai_internal/tests")
    return test_suite


if __name__ == '__main__':
    unittest.run(unittests())
