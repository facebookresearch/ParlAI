#!/bin/bash

PATH_TO_DATA="./projects/roscoe/roscoe_data"
RESTORE_SCRIPT="./projects/roscoe/roscoe_data/restore_annotated.py"

mkdir -p ${PATH_TO_DATA}/raw
mkdir -p ${PATH_TO_DATA}/generated
mkdir -p ${PATH_TO_DATA}/annotated

#Step 1. Download [dataset]_reasoning.txt in projects/roscoe/roscoe_data/generated/
echo "Sorry, Pending data release approval"
exit

#Step 2. Download original datasets in projects/roscoe/roscoe_data/raw/
# DROP
wget https://ai2-public-datasets.s3.amazonaws.com/drop/drop_dataset.zip
unzip drop_dataset.zip
mv drop_dataset/drop_dataset_dev.json ${PATH_TO_DATA}/raw/drop.txt
rm drop_dataset.zip
rm -rf drop_dataset
# Cosmos QA
wget https://github.com/wilburOne/cosmosqa/raw/master/data/valid.csv
mv valid.csv ${PATH_TO_DATA}/raw/cosmos.txt
# e-SNLI
wget https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_dev.csv
mv esnli_dev.csv ${PATH_TO_DATA}/raw/esnli.txt
# SemEval
wget https://raw.githubusercontent.com/DungLe13/commonsense/master/data/clean_data/clean-train-data.xml
mv clean-train-data.xml ${PATH_TO_DATA}/raw/semevalcommonsense.txt
# GSM8K
wget https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/example_model_solutions.jsonl
mv example_model_solutions.jsonl ${PATH_TO_DATA}/raw/gsm8k.txt

#Step 3. Restore sets
python ${RESTORE_SCRIPT}

#Step 4. Download annotated sets in projects/roscoe/roscoe_data/annotated/
echo "Sorry, Pending data release approval"
exit
