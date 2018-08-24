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
EXCLUDE_PATHS='^\(parlai_internal\|docs/build\)'

CR[0]="Copyright (c) 2017-present, Facebook, Inc."
CR[1]="All rights reserved."
CR[2]="This source code is licensed under the BSD-style license found in the"
cr[3]="LICENSE file in the root directory of this source tree. An additional grant"
CR[4]="of patent rights can be found in the PATENTS file in the same directory."

CR_LEN=${#CR[*]}  # number elements in the message

EXIT_CODE=0

FILES="$(find . -type f -iregex "$EXTENSIONS" -printf '%P\n' | grep -v "$EXCLUDE_PATHS")"

# for each code file
for fn in $FILES
do
    # skip empty files
    if [[ "$(wc -l $fn | cut -f1 -d ' ')" == "0" ]]
    then
        continue
    fi
    # check each line of the copyright is in the file
    for i in $(seq 0 $[CR_LEN - 1])
    do
        msg="${CR[$i]}"
        head -n "$LINE_LIMIT" "$fn" | grep "$msg" -Fc > /dev/null
        # if we didn't find anything, warn:
        if [ "$?" != "0" ]
        then
            EXIT_CODE=2
            echo "$fn:1: missing \"$msg\""
        fi
    done
done

exit "$EXIT_CODE"
