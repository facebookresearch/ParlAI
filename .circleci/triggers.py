#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Detects whether specific CircleCI jobs should run.

You can get CircleCI to run extra tests for your pull request by adding special
words into your commit messages.

[gpu]: Run the nightly GPU tests
[mturk]: Run the mturk tests
[long] or [all]: run all above
"""

import parlai.utils.testing as testing_utils


def detect_all():
    """
    Check if we should run all tests.
    """
    return any(
        kw in testing_utils.git_commit_messages().lower() for kw in ['[all]', '[long]']
    )


def detect_osx():
    """
    Check if we should run OSX tests.
    """
    commit_msg = '[osx]' in testing_utils.git_commit_messages().lower()
    return commit_msg


def detect_gpu():
    """
    Check if we should run GPU tests.
    """
    commit_msg = '[gpu]' in testing_utils.git_commit_messages().lower()
    test_changed = any(
        'tests/nightly/gpu' in fn for fn in testing_utils.git_changed_files()
    )
    return commit_msg or test_changed


MAPPING = {
    'nightly_gpu_tests': detect_gpu,
    'unittests_osx': detect_osx,
}


def main():
    """
    Run the program, printing the name of tests we should run to stdout.
    """
    run_all = detect_all()
    for testname, detector in MAPPING.items():
        if run_all or detector():
            print(testname)


if __name__ == '__main__':
    main()
