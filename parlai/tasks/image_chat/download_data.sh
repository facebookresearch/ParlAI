#!/usr/bin/env bash

export CURRENT_DIR=${PWD}
# Should return the ParlAI clone path
echo "Running code from: " $CURRENT_DIR

export DATA_DIR=/tmp/
mkdir -p $DATA_DIR
echo "Downloading in data root: " $DATA_DIR

PYTHONPATH=. python parlai/tasks/image_chat/download_data.py \
-dp $DATA_DIR
