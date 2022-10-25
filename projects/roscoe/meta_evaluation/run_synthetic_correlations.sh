#!/bin/bash

SCORER="./projects/roscoe/meta_evaluation/roscoe_synthetic_correlations.py"

python $SCORER --dataset-name aqua
python $SCORER --dataset-name asdiv
python $SCORER --dataset-name entailment_bank
python $SCORER --dataset-name eqasc
python $SCORER --dataset-name math_dataset
python $SCORER --dataset-name proofwriter
