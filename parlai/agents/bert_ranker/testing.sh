#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# download pretrained bert model if needed.
echo "Downloading required models if needed"
MODELFILE="./bert-base-uncased.tar.gz"
if [ ! -f $MODELFILE ]; then
    echo "Downloading the pretrained model"
    wget "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz" -O $MODELFILE
fi

VOCABFILE="./bert-base-uncased-vocab.txt"
if [ ! -f $VOCABFILE ]; then
    echo "Downloading the pretrained model"
    wget "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt" -O $VOCABFILE
fi
echo "All models are there"

# Train the bi-encoder
# 2 GPU machines: batchsize 32 num-epoch 0.1 warmup_updates 200. 25% accuracy in 5min (+ eval)
# 8 GPU machines: batchsize 128 num-epoch 3 warmup_updates 4000. 79% accuracy in 40mn
python -u parlai/scripts/train_model.py -pyt convai2 \
  -m parlai.agents.bert_ranker.biencoder_ranker:BiEncoderRankerAgent \
  --batchsize 32 --dict-file ./dictionary --model-file ./my_biencoder  \
  --pretrained-bert-path $MODELFILE --bert-vocabulary-path $VOCABFILE \
  --eval-batchsize 8 --learningrate 5e-5  --log_every_n_secs 10 \
  --shuffle true --type-optimization all_encoder_layers \
  --data-parallel true --history-size 5 --label-truncate 300 \
  --text-truncate 300 --num-epochs 0.1 \
  --lr-scheduler fixed --lr-scheduler-patience 1 --lr-scheduler-decay 0.35 \
  -veps 1.0 -vme 2000  --warmup_updates 200
# Should yield
# test:{'exs': 7801, 'accuracy': 0.25, 'f1': 0.3272, 'hits@1': 0.25,
#       'hits@5': 0.634, 'hits@10': 0.845, 'hits@100': 1.0, 'bleu': 0.2502,
#       'lr': 5e-05, 'num_updates': 411, 'examples': 7801, 'loss': 2388.0,
#       'mean_loss': 0.3062, 'mean_rank': 5.269}


# Train the cross-encoder
# 2 GPU machines: batchsize 2 num-epoch 0.01 (!!!!!) warmup_updates 100 65% accuracy in 5min (+ eval)
# 8 GPU machines: batchsize 8 num-epoch 3 warmup_updates 4000. 85.2% accuracy in 8 hours
python -u parlai/scripts/train_model.py -pyt convai2 \
  -m parlai.agents.bert_ranker.crossencoder_ranker:CrossEncoderRankerAgent \
  --batchsize 2 --dict-file ./dictionary --model-file ./my_crossencoder  \
  --pretrained-bert-path $MODELFILE --bert-vocabulary-path $VOCABFILE \
  --eval-batchsize 2 --learningrate 5e-5  --log_every_n_secs 10 \
  --shuffle true --type-optimization all_encoder_layers --lr-scheduler fixed \
  --lr-scheduler-patience 1 --lr-scheduler-decay 0.35 \
  --data-parallel true --history-size 5 --label-truncate 300 \
  --text-truncate 300 --num-epochs 0.01 \
  --lr-scheduler fixed --lr-scheduler-patience 1 --lr-scheduler-decay 0.35 \
  -veps 1.0 -vme 2000  --warmup_updates 100
# Should yield
# {'exs': 7801, 'accuracy': 0.653, 'f1': 0.6886, 'hits@1': 0.653,
# 'hits@5': 0.904, 'hits@10': 0.97, 'hits@100': 1.0, 'bleu': 0.653,
# 'lr': 5e-05, 'num_updates': 0, 'examples': 7801, 'loss': 13380.0,
#  'mean_loss': 1.715, 'mean_rank': 2.291}

# Evaluate a model that selects to top 10 elements with our previously
# trained biencoder and test it on the previously trained crossencoder
# takes about 20mn to evaluate on 2 GPU
python -u parlai/scripts/eval_model.py -pyt convai2 \
  -m parlai.agents.bert_ranker.bert_ranker:BothEncoderRankerAgent \
  --batchsize 2 --dict-file ./dictionary \
  --pretrained-bert-path $MODELFILE --bert-vocabulary-path $VOCABFILE \
  --shuffle true --log_every_n_secs 10 --lr_scheduler none\
  --data-parallel true --history-size 5 --label-truncate 300 \
  --text-truncate 300 \
  --biencoder-model-file ./my_biencoder \
  --crossencoder-model-file ./my_crossencoder
# Should yield
# {'exs': 7801, 'accuracy': 0.6208, 'f1': 0.6597, 'hits@1': 0.621,
# 'hits@5': 0.819, 'hits@10': 0.845, 'hits@100': 0.845, 'bleu': 0.6209,
# 'num_updates': 0}
# low score because at this point biencoder is terrible.

# uncomment if doing this locally: have a conversation with out model
# (note: conversation might be terrible with such a poorly trained biencoder)
# python examples/interactive.py \
#   -m parlai.agents.bert_ranker.bert_ranker:BothEncoderRankerAgent \
#   --pretrained-bert-path $MODELFILE --bert-vocabulary-path $VOCABFILE \
#   --fixed-candidates-path ~/data/pc_top_cands.txt \
#   --biencoder-model-file ./my_biencoder \
#   --crossencoder-model-file ./my_crossencoder \
#   --eval-candidates fixed --lr_scheduler none
# Enter Your Message: your persona:i love football\n Hi how are you?
# [TorchAgent]: hey there . how are you ?#
