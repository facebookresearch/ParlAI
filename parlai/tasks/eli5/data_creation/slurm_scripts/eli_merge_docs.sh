# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${DIR}/.."
pwd
echo $NM
echo $C
python merge_support_docs.py -n $NM -c $C
