#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#SBATCH --job-name=extract
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/merge_docs_%A.out
#SBATCH --error=logs/merge_docs_%A.err

./eli_merge_docs.sh
