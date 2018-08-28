#!/bin/bash

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# this file makes sure we're including the appropriate copyright message in all
# public code

# message must appear within the first few lines
LINE_LIMIT=20
# filenames to check
EXTENSIONS='.*\.\(rst\|py\|sh\|js\)$'
# don't check these files

CR[0]="Copyright (c) 2017-present, Facebook, Inc."
CR[1]="All rights reserved."
CR[2]="This source code is licensed under the BSD-style license found in the"
CR[3]="LICENSE file in the root directory of this source tree. An additional grant"
CR[4]="of patent rights can be found in the PATENTS file in the same directory."

CR_LEN=${#CR[*]}  # number elements in the message

EXIT_CODE=0

if [ "$1" == "all" ]
then
    # force a check on every file
    FILES="$(git ls-files | grep "$EXTENSIONS")"
else
    FILES="$(git diff --name-only origin/master | grep "$EXTENSIONS")"
fi

# for each code file
for fn in $FILES
do
    # skip files that don't exist in this repo (happens if we're behind master)
    if [ ! -f "$fn" ]
    then
        continue
    fi

    # skip empty files
    if [[ "$(wc -l $fn | cut -f1 -d ' ')" == "0" ]]
    then
        continue
    fi

    # skip items we don't own the copyright for
    grep -q 'Moscow Institute of Physics and Technology.' $fn && continue
    grep -q 'https://github.com/fartashf/vsepp' $fn && continue

    # check each line of the copyright is in the file
    for i in $(seq 0 $[CR_LEN - 1])
    do
        if [[ "$fn" =~ "mlb_vqa" ]] && [[ $i < 2 ]]
        then
            # special case for VQA
            continue
        fi
        msg="${CR[$i]}"
        head -n "$LINE_LIMIT" "$fn" | grep "$msg" -Fq
        # if we didn't find anything, warn:
        if [ "$?" != "0" ]
        then
            EXIT_CODE=2
            echo "$fn:1: missing \"$msg\""
        fi
    done
done

exit "$EXIT_CODE"
