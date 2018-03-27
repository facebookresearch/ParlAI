# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

python ~/ParlAI/examples/train_model.py --task convai2:self --model seq2seq -hs 1024 -esz 256 -att general -nl 2 -rnn lstm -lr 3 -dr 0.1 -clip 0.1 -lt enc_dec -opt sgd -emb glove -mom 0.9 -bi False -bs 128 -clen -1 --validation-every-n-secs 90 --validation-metric ppl --validation-metric-mode min --validation-patience 12 --save-after-valid True --dict-file /tmp/dict_convai2 --dict-lower True --dict-maxexs -1 --log-every-n-secs 10 --model-file /tmp/convai2_self_seq2seq_model
