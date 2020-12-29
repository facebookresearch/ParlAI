#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import mephisto


TASK_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(mephisto.__file__))),
    'examples',
    'parlai_chat_task_demo',
)
