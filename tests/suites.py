#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Various test loaders.
"""

import os
import unittest
import random
import time
from itertools import chain


def _circleci_parallelism(suite):
    """
    Allow for parallelism in CircleCI for speedier tests..
    """
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


def datatests():
    """
    Test for data integrity.

    Runs on CircleCI. Separate to help distinguish failure reasons.
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests/datatests')
    return test_suite


def nightly_gpu():
    """
    Nightly GPU tests.

    Runs on CircleCI nightly, and when [gpu] or [long] appears in a commit string.
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests/nightly/gpu')
    test_suite = _circleci_parallelism(test_suite)
    return test_suite


def unittests():
    """
    Short tests.

    Runs on CircleCI on every commit. Returns everything in the tests root directory.
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    test_suite = _circleci_parallelism(test_suite)
    return test_suite


def mturk():
    """
    Mechanical Turk tests.
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("parlai/mturk/core/test/")
    return test_suite


def internal_tests():
    """
    Internal Tests.
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("parlai_internal/tests")
    return test_suite


class TimeLoggingTestResult(unittest.runner.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_timings = []

    def startTest(self, test):
        self._test_started_at = time.time()
        super().startTest(test)

    def addSuccess(self, test):
        elapsed = time.time() - self._test_started_at
        name = self.getDescription(test)
        self.test_timings.append((name, elapsed))
        super().addSuccess(test)

    def getTestTimings(self):
        return self.test_timings


class TimeLoggingTestRunner(unittest.TextTestRunner):
    def __init__(self, slow_test_threshold=10.0, *args, **kwargs):
        self.slow_test_threshold = slow_test_threshold
        return super().__init__(resultclass=TimeLoggingTestResult, *args, **kwargs)

    def run(self, test):
        result = super().run(test)

        self.stream.writeln("\nSlow Tests (>{:.03}s):".format(self.slow_test_threshold))
        for name, elapsed in result.getTestTimings():
            if elapsed > self.slow_test_threshold:
                self.stream.writeln("({:.03}s) {}".format(elapsed, name))

        return result


MySuite = unittests()


def main():
    unittest.main(testRunner=TimeLoggingTestRunner)


if __name__ == '__main__':
    main()
