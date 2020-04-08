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
from pytest import ExitCode


# TODO: rename the folders nicer so they make more sense, maybe even have
# a 1:1 correspondance with the circleci name


def pytest_collection_modifyitems(config, items):
    # handle circleci parallelism
    if 'CIRCLE_NODE_TOTAL' in os.environ:
        total = int(os.environ['CIRCLE_NODE_TOTAL'])
        index = int(os.environ['CIRCLE_NODE_INDEX'])
    else:
        total = 1
        index = 0

    # python 3.4/3.5 compat: rootdir = pathlib.Path(str(config.rootdir))
    rootdir = pathlib.Path(config.rootdir)
    parallels = [i % total == index for i in range(len(items))]
    random.Random(42).shuffle(parallels)
    deselected = []
    for parallel, item in zip(parallels, items):
        rel_path = str(pathlib.Path(item.fspath).relative_to(rootdir))
        if not parallel:
            deselected.append(item)
        elif "parlai_internal" in rel_path:
            item.add_marker("internal")
        elif "nightly/gpu/" in rel_path:
            item.add_marker("nightly_gpu")
        elif "nightly/cpu/" in rel_path:
            item.add_marker("nightly_cpu")
        elif "datatests/" in rel_path:
            item.add_marker("data")
        elif "tasks/" in rel_path:
            item.add_marker("tasks")
        elif "parlai/mturk/core/test/" in rel_path:
            item.add_marker("mturk")
        elif "/" not in rel_path[6:]:
            item.add_marker("unit")
        else:
            raise ValueError(f"Couldn't categorize '{rel_path}'")
    config.hook.pytest_deselected(items=deselected)
    for d in deselected:
        items.remove(d)


def pytest_sessionfinish(session, exitstatus):
    """
    Ensure that pytest doesn't report failure when no tests are collected.

    This can sometimes happen due to the way we distribute tests across multiple circle
    nodes.
    """
    if exitstatus == ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = ExitCode.OK
