#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class LaunchConfig:
    """
    Use this file to put all the arguments (an alternative to launching them from the
    command line as params)
    """

    BLUEPRINT_TYPE = 'turn_annotations_static_inflight_qa_blueprint'

    # Run the below to register the requester if needed (first time only)
    # mephisto register mturk_sandbox --name:XXXX
    # --access-key-id:XXXX --secret-access-key:XXXX
    # To run locally, set REQUEST = None and PROVIDER = 'mock'
    REQUESTER = 'FIXME'
    PROVIDER = 'mturk_sandbox'

    # This datapath is where the database object goes
    # If not Mephisto data path below then requester register seems to do nothing
    DATAPATH = 'FIXME'

    TASK_TITLE = 'Chat with a fellow conversationalist'
    TASK_DESCRIPTION = '''<br>
    <b><h4>Task Description</h4></b>
    <br>
    Dummy Task Description.

    Lorem ipsum.
    <br><br>
      '''

    FILE_DATA_JSONL = 'FIXME'

    TASK_REWARD = 0.3
    SUBTASKS_PER_UNIT = 6

    # How many workers to do each assignment
    UNITS_PER_ASSIGNMENT = 5

    # Maximum tasks a worker can do across all runs with task_name (0=infinite)
    MAX_UNITS_PER_WORKER = 5

    # A list of worker IDs to block from doing the task.
    WORKER_BLOCK_LIST = []

    # Blueprint specific params
    ASK_REASON = False
    ANNOTATION_INDICES_JSONL = None
