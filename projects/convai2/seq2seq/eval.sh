# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

if [ ! -f /tmp/convai2_self_seq2seq_model ]; then
	echo "Downloading model to /tmp/convai2_self_seq2seq_model. Should get 29.73 validation ppl."
	wget https://s3.amazonaws.com/fair-data/parlai/_models/convai2/convai2_self_seq2seq_model -O /tmp/convai2_self_seq2seq_model 
fi
python ~/ParlAI/examples/eval_model.py -t convai2:self -dt valid -bs 128 -m seq2seq -mf /tmp/convai2_self_seq2seq_model 
