# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
python ../../examples/eval_model.py -m retriever_reader --reader-model-file /tmp/model_drqa --retriever-model-file "models:wikipedia_full/tfidf_retriever/model" -t squad:openSquad
