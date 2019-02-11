#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
