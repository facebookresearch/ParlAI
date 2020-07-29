# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
parlai eval_model -m retriever_reader --reader-model-file /tmp/model_drqa --retriever-model-file "models:wikipedia_full/tfidf_retriever/model" -t squad:openSquad
