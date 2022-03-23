#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
THIS_VARIANT = "multiwoz"

LOTS_NUCLEUS09_TOKENEM = "lotsNuc09tokenEm"

NUCLEUS_VALS = {
    LOTS_NUCLEUS09_TOKENEM: " ".join(["0.9"] * 20),
}


SWEEP_SHORT_NAME = {
    LOTS_NUCLEUS09_TOKENEM: "lotsNuc09tokenEm",
}

LADDER_LOOKUP = {}

################ ITER CUMULATIVE
LOTS_NUCLEUS09_TOKENEM_FOLDERS = [

]
LOTS_NUCLEUS09_TOKENEM_CONVO_LADDER = [
    x + "/concatinated.jsonl" for x in LOTS_NUCLEUS09_TOKENEM_FOLDERS
]

LADDER_LOOKUP[LOTS_NUCLEUS09_TOKENEM] = LOTS_NUCLEUS09_TOKENEM_CONVO_LADDER
