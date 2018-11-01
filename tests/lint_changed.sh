#!/bin/sh

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# This shell script lints only the things that changed in the most recent change.
# It's much more strict than our check for lint across the entire code base.

set -e
flake8 --version | grep '^3\.6\.' >/dev/null || \
    ( echo "Please install flake8 >=3.6.0." && false )

CHANGED_FILES="$(git diff --name-only master... | grep '\.py$' | tr '\n' ' ')"
if [ "$CHANGED_FILES" != "" ]
then
    exec flake8 $CHANGED_FILES
fi
