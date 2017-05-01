#!/bin/bash

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# if parlai can't be imported after running this, try running this code from
# your terminal directly instead of running this file--sometimes this won't work
# properly if you have python aliased

if [ -e requirements.txt ]; then
    pip install -r requirements.txt
fi

python setup.py develop
