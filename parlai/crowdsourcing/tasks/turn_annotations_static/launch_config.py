#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class LaunchConfig:
    """
    Use this file to put all the arguments (an alternative to launching them from the
    command line as params)
    """

    TASK_TITLE = 'Chat with a fellow conversationalist'
    TASK_DESCRIPTION = '''<br>
    <b><h4>Task Description</h4></b>
    <br>
    Dummy Task Description.

    Lorem ipsum.
    <br><br>
      '''

    DATA_JSONL = 'FIXME'

    TASK_REWARD = 0.3

    # How many workers to do each assignment
    UNITS_PER_ASSIGNMENT = 5

    # Maximum tasks a worker can do across all runs with task_name (0=infinite)
    MAXIMUM_UNITS_PER_WORKER = 5

    # A list of worker IDs to block from doing the task.
    WORKER_BLOCK_LIST = []
