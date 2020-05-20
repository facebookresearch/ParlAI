#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

([ ! -z "$CIRCLE_NODE_INDEX" ] && [ "$CIRCLE_NODE_INDEX" != 2 ]) && echo "SKIPPED" && exit

set -e -x  # error and exit on any failure; print the commands being run

# view a task & train a model
python examples/display_data.py -t babi:task10k:1
python examples/train_model.py -t babi:task10k:1 -mf /tmp/babi_memnn -bs 1 -nt 4 -eps 5 -m memnn --no-cuda
python examples/display_model.py -t babi:task10k:1 -mf /tmp/babi_memnn -ecands vocab

# train a transformer on twitter
pip3 install emoji unidecode
python examples/display_data.py -t twitter
python examples/train_model.py -t twitter -mf /tmp/tr_twitter -m transformer/ranker -bs 10 -vtim 3600 -cands batch -ecands batch --data-parallel True --max-train-time 20 -nl 1 --dict-tokenizer split -emb random --ffn-size 128
python examples/eval_model.py -t twitter -bs 50 -m legacy:seq2seq:0 -mf models:convai2/seq2seq/convai2_self_seq2seq_model --num-examples 1
python examples/display_model.py -t twitter -mf /tmp/tr_twitter -ecands batch

# add a simple model
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

python examples/display_model.py -t babi:task10k:1 -m parrot
python examples/build_dict.py -t babi:task10k:1 -df /tmp/parrot.dict
python examples/display_model.py -t babi:task10k:1 -m parrot -df /tmp/parrot.dict
