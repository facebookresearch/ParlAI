#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

([ ! -z "$CIRCLE_NODE_INDEX" ] && [ "$CIRCLE_NODE_INDEX" != 2 ]) && echo "SKIPPED" && exit

set -e -x  # error and exit on any failure; print the commands being run

# test installation & help commands
parlai
parlai help > /dev/null
parlai train --help | grep -- --task > /dev/null

# view a task & train a model
parlai display_data -t babi:task10k:1
parlai train_model -t babi:task10k:1 -mf /tmp/babi_memnn -bs 1 -eps 2 -m memnn --no-cuda
parlai display_model -t babi:task10k:1 -mf /tmp/babi_memnn -ecands vocab

# train a transformer on twitter
rm -rf /tmp/tr_twitter*
python -m pip install emoji unidecode
parlai display_data -t twitter
parlai train_model -t twitter -mf /tmp/tr_twitter -m transformer/ranker -bs 10 -vtim 3600 -cands batch -ecands batch --data-parallel True --max-train-time 20 -nl 1 --dict-tokenizer split -emb random --ffn-size 128
parlai eval_model -t twitter -bs 32 -mf zoo:blender/blender_90M/model --num-examples 1
parlai display_model -t twitter -mf /tmp/tr_twitter -ecands batch

# add a simple model
rm -rf /tmp/parrot*
rm -rf parlai/agents/parrot
mkdir parlai/agents/parrot
touch parlai/agents/parrot/parrot.py

cat > parlai/agents/parrot/parrot.py <<- EOF
from parlai.core.torch_agent import TorchAgent, Output
class ParrotAgent(TorchAgent):
    def train_step(self, batch):
        pass

    def eval_step(self, batch):
        return Output([self.dict.vec2txt(row) for row in batch.text_vec])

    def build_model(self):
        return None
EOF

rm -rf /tmp/babi_memnn*
parlai display_model -t babi:task10k:1 -m parrot
parlai build_dict -t babi:task10k:1 -df /tmp/parrot.dict
parlai display_model -t babi:task10k:1 -m parrot -df /tmp/parrot.dict
