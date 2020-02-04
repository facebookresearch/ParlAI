#!/usr/bin/env bash

export CURRENT_DIR=${PWD}
# Should ideally give the parlai clone path
echo "Running code from: " $CURRENT_DIR

#export TASK_DIR="$(dirname "$CURRENT_DIR")"
#export PARLAI_CODE_DIR="$(dirname "$TASK_DIR")"
#export PROJECT_DIR="$(dirname "$PARLAI_CODE_DIR")"
# Going to the project directory
#cd $PROJECT_DIR

export DATA_DIR=/tmp/
mkdir -p $DATA_DIR

echo "Downloading in data root: " $DATA_DIR

PYTHONPATH=. python parlai/tasks/image_chat/download_data.py \
-dp $DATA_DIR
