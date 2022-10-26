#!/bin/bash

# Example:
# sh projects/roscoe/synthetic_evaluation/score_all.sh sim_sce facebook/roscoe-512-roberta-base

print_usage() {
  printf "Usage: score_all.sh model_type model_name"
}

if [ $# -ne 2 ] ; then
    echo "Require 2 arguments: model type, and model name"
    print_usage
    exit 1
fi

SCORER="./projects/roscoe/synthetic_evaluation/synthetic_roscoe.py"

python $SCORER -t $1 -m $2 -p ./projects/roscoe/roscoe_data/synthetic_50%/aqua_synthetic/ -db 256 -cb 64
python $SCORER -t $1 -m $2 -p ./projects/roscoe/roscoe_data/synthetic_50%/asdiv_synthetic/ -db 256 -cb 64
python $SCORER -t $1 -m $2 -p ./projects/roscoe/roscoe_data/synthetic_50%/entailment_bank_synthetic/ -db 256 -cb 64
python $SCORER -t $1 -m $2 -p ./projects/roscoe/roscoe_data/synthetic_50%/eqasc_synthetic/ -db 256 -cb 64
python $SCORER -t $1 -m $2 -p ./projects/roscoe/roscoe_data/synthetic_50%/math_dataset_synthetic/ -db 32 -cb 32
python $SCORER -t $1 -m $2 -p ./projects/roscoe/roscoe_data/synthetic_50%/proofwriter_synthetic/ -db 256 -cb 64
