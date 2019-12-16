#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class DefaultTeacher:
    # Dummy class.
    pass


class InteractiveTeacher:
    # Dummy class to add arguments for interactive world.
    pass


def create_agents(opt):
    # interactive task has no task agents (they are attached as user agents)
    return []
