#!/bin/sh

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This shell script lints only the things that changed in the most recent change.

set -e

onlyexists() {
    if [[ "$1" != "" ]]; then
        PRE="$1/"
    else
        PRE=""
    fi
    while read fn; do
        if [ -f "${PRE}${fn}" ]; then
            echo "$fn"
        fi
    done
}

CMD="flake8"
CHANGED_FILES="$(git diff --name-only master... | grep '\.py$' | onlyexists | tr '\n' ' ')"
while getopts bi opt; do
  case $opt in
    i)
      ROOT="$(git -C ./parlai_internal/ rev-parse --show-toplevel)"
      CHANGED_FILES="$(git -C ./parlai_internal/ diff --name-only master... | onlyexists $ROOT |
      xargs -I '{}' realpath --relative-to=. $ROOT/'{}' |
      grep '\.py$' | tr '\n' ' ')"
      ;;
    b)
      CMD="black"
  esac

  done

if [ "$CHANGED_FILES" != "" ]
then
    if [[ "$CMD" == "black" ]]
    then
        command -v black >/dev/null || \
            ( echo "Please install black." && false )
        # only output if something needs to change
        black --check $CHANGED_FILES
    else
        flake8 --version | grep '^3\.[6-9]\.' >/dev/null || \
            ( echo "Please install flake8 >=3.6.0." && false )

        # soft complaint on too-long-lines
        flake8 --select=E501 --show-source $CHANGED_FILES
        # hard complaint on really long lines
        exec flake8 --max-line-length=127 --show-source $CHANGED_FILES
    fi
fi
