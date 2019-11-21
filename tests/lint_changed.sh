#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This shell script lints only the things that changed in the most recent change.


DOCOPTS="--pre-summary-newline --wrap-descriptions 88 --wrap-summaries 88 --make-summary-multi-line"

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

RUNALL=1
INTERNAL=0
CHECK=0
CMD=""
CHANGED_FILES="$(git diff --name-only master... | grep '\.py$' | onlyexists | tr '\n' ' ')"
while getopts bidcf opt; do
  case $opt in
    f)
      [[ "$CMD" != "" ]] && (echo "Don't mix args." && false);
      RUNALL=0
      CMD="flake8"
      ;;
    c)
      CHECK=1
      ;;
    d)
      [[ "$CMD" != "" ]] && (echo "Don't mix args." && false);
      CMD="docformatter"
      RUNALL=0
      ;;
    i)
      INTERNAL=1
      ROOT="$(git -C ./parlai_internal/ rev-parse --show-toplevel)"
      CHANGED_FILES="$(git -C ./parlai_internal/ diff --name-only master... | onlyexists $ROOT |
      xargs -I '{}' realpath --relative-to=. $ROOT/'{}' |
      grep '\.py$' | tr '\n' ' ')"
      ;;
    b)
      [[ "$CMD" != "" ]] && (echo "Don't mix args." && false);
      CMD="black"
      RUNALL=0
      ;;
  esac
done

if [[ $RUNALL -eq 1 ]]
then
    if [[ $CHECK -eq 1 ]]; then A="$A -c"; fi
    if [[ $INTERNAL -eq 1 ]]; then A="$A -i"; fi
    bash $0 -b $A
    bash $0 -d $A
    bash $0 -f $A
    exit 0
fi

if [ "$CHANGED_FILES" != "" ]
then
    if [[ "$CMD" == "black" ]]
    then
        command -v black >/dev/null || \
            ( echo "Please install black." && false )
        # only output if something needs to change
        if [[ $FIXIT ]]
        then
            black $CHANGED_FILES
        else
            black --check $CHANGED_FILES
        fi
    elif [[ "$CMD" == "docformatter" ]]
    then
        command -v docformatter > /dev/null || \
            ( echo "Please run \`pip install docformatter\`." && false )
        if [[ $FIXIT ]]
        then
            docformatter -i $DOCOPTS $CHANGED_FILES
        else
            docformatter -c $DOCOPTS $CHANGED_FILES
        fi
    elif [[ "$CMD" == "flake8" ]]
    then
        flake8 --version | grep '^3\.[6-9]\.' >/dev/null || \
            ( echo "Please install flake8 >=3.6.0." && false )

        # soft complaint on too-long-lines
        flake8 --select=E501 --show-source $CHANGED_FILES
        # hard complaint on really long lines
        exec flake8 --max-line-length=127 --show-source $CHANGED_FILES
    else
        echo "Don't know how to \`$CMD\`."
        false
    fi
fi
