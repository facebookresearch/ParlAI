#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

export CURRENT_DIR=${PWD}
# Should return the ParlAI clone path
echo "Running code from: " $CURRENT_DIR

export DATA_DIR=/tmp/
mkdir -p $DATA_DIR
echo "Downloading in data root: " $DATA_DIR

PYTHONPATH=. python parlai/tasks/image_chat/download_data.py \
-dp $DATA_DIR
