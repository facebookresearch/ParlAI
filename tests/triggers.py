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

MAPPING = {
    'gpu': detect_gpu,
    'longcpu': detect_long_cpu,
    'newtasks': detect_new_tasks
}


def detect_gpu():
    test_changed = any(
        'tests/nightly/gpu' in fn
        for fn in testing_utils.git_changed_files()
    )
    commit_msg = any(
        '[gpu]' in msg
        for msg in testing_utils.git_commit_messages()
    )


def detect_long_cpu():
    pass


def detect_new_tasks():
    pass


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MAPPING:
        raise ValueError(
            'You must give exactly one argument of {{{}}}}.'.format
            ','.join(MAPPING.keys())
        )
    detector = MAPPING[sys.argv[1]]
    # if the detector returns true, we'll want to run that test
    if detector():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
