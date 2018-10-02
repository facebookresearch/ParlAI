#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import unittest

# gross hack b/c we're not marking tests/ as a package
try:
    from tests import utils
except ImportError:
    from . import utils


def _clear_cmdline_args(fn):
    """
    Helper decorator to make sure 'python setup.py test' doesn't look like
    a parlai call.
    """
    import sys
    sys.argv = sys.argv[:1]
    return fn


@_clear_cmdline_args
def integration():
    """Integration tests. Longer, and may need a GPU."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests/integration')
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
def travis():
    """Tests needed to pass Travis."""
    test_suite = unittest.TestSuite()
    test_suite.addTests(short())
    changed_files = utils.git_changed_files(skip_nonexisting=False)
    if any('parlai/mturk' in fn for fn in changed_files):
        # if any mturk stuff changed, run those tests too
        test_suite.addTests(mturk())
    return test_suite


@_clear_cmdline_args
def full():
    """All tests."""
    test_suite = unittest.TestSuite()
    test_suite.addTests(short())
    test_suite.addTests(integration())
    test_suite.addTests(mturk())
    return test_suite


if __name__ == '__main__':
    unittest.run(travis())
