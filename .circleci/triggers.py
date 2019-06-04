#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Detects whether specific CircleCI jobs should run.

You can get CircleCI to run extra tests for your pull request by adding special
words into your commit messages.

[gpu]: Run the nightly GPU tests
[mturk]: Run the mturk tests
[data]: Run the data tests
[long]: run all above
"""

import parlai.core.testing_utils as testing_utils


def detect_all():
    """Check if we should run all tests."""
    return '[long]' in testing_utils.git_commit_messages()


def detect_gpu():
    """Check if we should run GPU tests."""
    commit_msg = '[gpu]' in testing_utils.git_commit_messages()
    test_changed = any(
        'tests/nightly/gpu' in fn
        for fn in testing_utils.git_changed_files()
    )
    return commit_msg or test_changed


def detect_data():
    """Check if we should run data tests."""
    commit_msg = '[data]' in testing_utils.git_commit_messages().lower()
    test_changed = any(
        testing_utils.is_new_task_filename(fn)
        for fn in testing_utils.git_changed_files()
    )
    return commit_msg or test_changed


def detect_mturk():
    """Check if we should run mturk tests."""
    commit_msg = '[mturk]' in testing_utils.git_commit_messages().lower()
    mturk_changed = any(
        'parlai/mturk' in fn
        for fn in testing_utils.git_changed_files()
    )
    return commit_msg or mturk_changed


MAPPING = {
    'nightly_gpu_tests': detect_gpu,
    'datatests': detect_data,
    'mturk_tests': detect_mturk,
}


def main():
    """Run the program, printing the name of tests we should run to stdout."""
    run_all = detect_all()
    for testname, detector in MAPPING.items():
        if run_all or detector():
            print(testname)


if __name__ == '__main__':
    main()
