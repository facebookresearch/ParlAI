# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Trains a generative model on personachat using the 'self' profile
# Run from ParlAI directory

python examples/train_model.py -m "projects.personachat.persona_seq2seq:PersonachatSeqseqAgentSplit"  -t personachat:self -bs 128 -rc True --validation-every-n-secs 300 -vp 100 -lr 0.001 -mf /tmp/profilememory_pc -enc lstm -hs 1024 -nl 1 --dropout 0.2 --attention general --personachat_attnsentlevel --personachat_sharelt --personachat_reweight 'use' --personachat_learnreweight -tr 100
