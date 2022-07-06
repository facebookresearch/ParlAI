# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${DIR}/.."
pwd
echo $SCNUM
python download_support_docs.py -ns $SCNUM
