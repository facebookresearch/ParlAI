#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Detects whether specific CircleCI jobs should run.
"""

import parlai.core.testing_utils as testing_utils


def detect_all():
    return '[long]' in testing_utils.git_commit_messages()


def detect_gpu():
    commit_msg = '[gpu]' in testing_utils.git_commit_messages()
    test_changed = any(
        'tests/nightly/gpu' in fn
        for fn in testing_utils.git_changed_files()
    )
    return commit_msg or test_changed


def detect_data():
    commit_msg = '[data]' in testing_utils.git_commit_messages().lower()
    test_changed = any(
        testing_utils.is_new_task_filename(fn)
        for fn in testing_utils.git_changed_files()
    )
    return commit_msg or test_changed


def detect_mturk():
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
    run_all = detect_all()
    for testname, detector in MAPPING.items():
        if run_all or detector():
            print(testname)


if __name__ == '__main__':
    main()
