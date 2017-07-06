#!/bin/bash

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

set -e # stop if any tests fail
python3 test_init.py
python3 test_import.py
python3 test_dict.py
python3 test_tasklist.py
python3 test_threadutils.py
python3 test_utils.py
