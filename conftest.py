#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This is used to configure markers on tests based on filename.

We use this to split up tests into different circleci runs.
"""

import os
import pathlib
import random
import collections
import pytest
import subprocess


# TODO: rename the folders nicer so they make more sense, maybe even have
# a 1:1 correspondance with the circleci name


# -----------------------------------------------------------------------
# From https://github.com/ryanwilsonperkin/pytest-circleci-parallelized.
# MIT licensed, Copyright Ryan Wilson-Perkin.
# -----------------------------------------------------------------------
def get_class_name(item):
    class_name, module_name = None, None
    for parent in reversed(item.listchain()):
        if isinstance(parent, pytest.Class):
            class_name = parent.name
        elif isinstance(parent, pytest.Module):
            module_name = parent.module.__name__
            break

    # heuristic:
    # - better to group gpu and task tests, since tests from those modules
    #   are likely to share caching more
    # - split up the rest by class name because slow tests tend to be in
    #   the same module
    if class_name and '.tasks.' not in module_name:
        return "{}.{}".format(module_name, class_name)
    else:
        return module_name


def filter_tests_with_circleci(test_list):
    circleci_input = "\n".join(test_list).encode("utf-8")
    p = subprocess.Popen(
        ["circleci", "tests", "split"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    circleci_output, _ = p.communicate(circleci_input)
    return [
        line.strip() for line in circleci_output.decode("utf-8").strip().split("\n")
    ]


# -----------------------------------------------------------------------
MARKER_RULES = [
    ('parlai_internal', 'internal'),
    ('crowdsourcing/', 'crowdsourcing'),
    ('nightly/gpu', 'nightly_gpu'),
    ('nightly/cpu/', 'nightly_cpu'),
    ('datatests/', 'data'),
    ('parlai/tasks/', 'teacher'),
    ('tasks/', 'tasks'),
    ('tod/', 'tod'),
]


def pytest_collection_modifyitems(config, items):
    marker_expr = config.getoption('markexpr')

    deselected = []

    # first add all the markers, possibly filtering
    # python 3.4/3.5 compat: rootdir = pathlib.Path(str(config.rootdir))
    rootdir = pathlib.Path(config.rootdir)
    for item in items:
        rel_path = str(pathlib.Path(item.fspath).relative_to(rootdir))
        for file_pattern, marker in MARKER_RULES:
            if file_pattern in rel_path:
                item.add_marker(marker)
                if marker_expr and marker != marker_expr:
                    deselected.append(item)
                break
        else:
            assert "/" not in rel_path[6:], f"Couldn't categorize '{rel_path}'"
            item.add_marker("unit")
            if marker_expr not in ['', 'unit']:
                deselected.append(item)

    # kill everything that wasn't grabbed
    for item in deselected:
        items.remove(item)

    if 'CIRCLE_NODE_TOTAL' in os.environ:
        # circleci, split up the parallelism by classes
        class_mapping = collections.defaultdict(list)
        for item in items:
            class_name = get_class_name(item)
            class_mapping[class_name].append(item)

        test_groupings = list(class_mapping.keys())
        random.Random(1339).shuffle(test_groupings)

        filtered_tests = filter_tests_with_circleci(test_groupings)
        new_items = []
        for name in filtered_tests:
            new_items.extend(class_mapping[name])
            items[:] = new_items


def pytest_sessionfinish(session, exitstatus):
    """
    Ensure that pytest doesn't report failure when no tests are collected.

    This can sometimes happen due to the way we distribute tests across multiple circle
    nodes.
    """
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
