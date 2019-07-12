#!/bin/sh

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This shell script lints only the things that changed in the most recent change.
# It's much more strict than our check for lint across the entire code base.

set -e

CMD="flake8"
CHANGED_FILES="$(git diff --name-only master... | grep '\.py$' | tr '\n' ' ')"
while getopts bi opt; do
  case $opt in
    i)
      CHANGED_FILES="$(git -C ./parlai_internal/ diff --name-only master... |
      xargs -I '{}' realpath --relative-to=. $(git -C ./parlai_internal/ rev-parse --show-toplevel)/'{}' |
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
