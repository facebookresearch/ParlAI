#!/bin/bash
#SBATCH --job-name=scratch_multiwoz2.3
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/project/ParlAI/bash_scripts/slurm_logs/scratch_multiwoz2.3-%j.log


#PFT if needed 


#FT 


# evaluate


# organize evaluation results 