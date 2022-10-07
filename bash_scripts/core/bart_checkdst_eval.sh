#!/bin/bash
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id

# CMD=$1
# echo $CMD 
# eval $CMD