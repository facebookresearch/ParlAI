#!/bin/bash

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Uses coverage.py

pip install coverage
if [[ ! -f setup.py ]]; then
    cd ..
fi
coverage run --include "parlai*" setup.py test

COVERED=""  # all tests with some unit tests

for line in $(coverage report); do
    if [[ $line =~ ".py" ]]; then
        COVERED="$COVERED;$line"
    fi
done

NOT_COV=""

for file in $(find parlai -name "*.py"); do
    if [[ ! $file =~ "__init__.py" ]]; then
        if [[ ! $COVERED =~ $file ]]; then
            file="0% coverage (not tested at all): $file"
            NOT_COV="$NOT_COV;$file"
        fi
    fi
done

echo $NOT_COV | tr ';' '\n'
echo
coverage report
