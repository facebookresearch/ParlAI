# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${DIR}/.."
pwd
echo $NM
echo $C
echo $NS
echo $NC
python select_sentences_tfidf.py -sr_n $NM -sid $C -ns $NS -nc $NC
