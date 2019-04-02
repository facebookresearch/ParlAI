#!/usr/bin/env python

"""
Detects whether specific jobs should run.

Current use cases:
    - Indicating there should be a hold marking a data change
    - Running nightly gpu tests if someone makes a change to those
    - Running nightly cpu tests if someone makes a change to those
"""

import sys
import parlai.core.testing_utils as testing_utils


def detect_gpu():
    commit_msg = '[gpu]' in testing_utils.git_commit_messages()
    test_changed = any(
        'tests/nightly/gpu' in fn
        for fn in testing_utils.git_changed_files()
    )
    return commit_msg or test_changed


def detect_long_cpu():
    commit_msg = '[longcpu]' in testing_utils.git_commit_messages().lower()
    test_changed = any(
        'tests/nightly/cpu' in fn
        for fn in testing_utils.git_changed_files()
    )
    return commit_msg or test_changed


def detect_data():
    commit_msg = '[newtask]' in testing_utils.git_commit_messages().lower()
    test_changed = any(
        testing_utils.check_new_task_filename(fn)
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
    'gpu': detect_gpu,
    'longcpu': detect_long_cpu,
    'data': detect_data,
    'mturk': detect_mturk,
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MAPPING:
        raise ValueError(
            'You must give exactly one argument of {{{}}}.'
            .format(','.join(MAPPING.keys()))
        )
    detector = MAPPING[sys.argv[1]]
    # if the detector returns true, we'll want to run that test
    if detector():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
