#!/bin/bash

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

set -e # stop if any tests fail
for test in $(ls); do
    if [ ${test: -3} == ".py" ] && [ $test != "test_downloads.py" ] && [ $test != "test_mlb_vqa.py" ]; then
        python3 $test;
    fi
done
