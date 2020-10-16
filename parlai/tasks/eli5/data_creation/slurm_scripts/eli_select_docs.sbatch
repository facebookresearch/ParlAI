#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#SBATCH --job-name=extract
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/select_docs_%A.out
#SBATCH --error=logs/select_docs_%A.err

./eli_select_docs.sh
