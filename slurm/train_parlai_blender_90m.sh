#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:3
#SBATCH --mem=48g  # Memory
#SBATCH --cpus-per-task=12  # number of cpus to use - there are 32 on each node.

# Set EXP_BASE_NAME and BATCH_FILE_PATH

echo "============"
echo "Initialize Env ========"

set -e # fail fast

export CURRENT_TIME=$(date "+%Y_%m_%d_%H%M%S")

# Activate Conda
source /home/${USER}/miniconda3/bin/activate ParlAI

echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Experiment name: ${EXP_NAME}"
dt=$(date '+%d_%m_%y__%H_%M')
echo ${dt}

# Env variables
export STUDENT_ID=${USER}

# General training parameters
export CLUSTER_HOME="/home/${STUDENT_ID}"
export DATASET_SOURCE="${CLUSTER_HOME}/datasets/ParlAI/"

declare -a ScratchPathArray=(/disk/scratch_big/ /disk/scratch1/ /disk/scratch2/ /disk/scratch/ /disk/scratch_fast/)

# Iterate the string array using for loop
for i in "${ScratchPathArray[@]}"; do
  echo ${i}
  if [ -d ${i} ]; then
    export SCRATCH_HOME="${i}/${STUDENT_ID}"
    mkdir -p ${SCRATCH_HOME}
    break
  fi
done

# Delete all scratch home sub directories more than a week old.
find ${SCRATCH_HOME} -type d -name "*" -mtime +7 -printf "%T+ %p\n" | sort | cut -d ' ' -f 2- | sed -e 's/^/"/' -e 's/$/"/' | xargs rm -rf

echo "Scratch home: ${SCRATCH_HOME}"

export EXP_ROOT="${CLUSTER_HOME}/git/ParlAI"

export EXP_ID="${EXP_NAME}_${SLURM_JOB_ID}_${CURRENT_TIME}"
echo "Experiment ID: ${EXP_ID}"
export SERIAL_DIR="${SCRATCH_HOME}/${EXP_ID}"
echo "Serial dir: ${SERIAL_DIR}"

cd "${EXP_ROOT}"

mkdir -p ${SERIAL_DIR}

echo "============"
echo "ParlAI Task========"

parlai train_model -t blended_skill_talk,wizard_of_wikipedia,convai2:normalized,writing_prompts \
-m transformer/generator --multitask-weights 1,3,3,3,10 --init-model zoo:tutorial_transformer_generator/model \
 --dict-file zoo:tutorial_transformer_generator/model.dict --embedding-size 512 --n-layers 8 --ffn-size 2048 \
 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm \
 --activation gelu --skip-generation True --fp16 True --text-truncate 512 --label-truncate 128 \
 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adamax --lr-scheduler reduceonplateau \
 --gradient-clip 0.1 -veps 0.25 --betas 0.9,0.999 --update-freq 1 --attention-dropout 0.0 --relu-dropout 0.0 \
 --skip-generation True --save-after-valid=True --num-epochs=50 --max_train_time=604800 --validation-patience=5 \
 --validation-every-n-secs=14400 --validation-max-exs=10000 --validation_every_n_epochs=-1 --batchsize=8 \
 --validation_metric=ppl --validation_metric_mode=min \
 -mf "${SERIAL_DIR}/${EXP_ID}.mod"

echo "============"
echo "ParlAI Task finished"

export HEAD_EXP_DIR="${CLUSTER_HOME}/runs/${EXP_ID}"
mkdir -p "${HEAD_EXP_DIR}"
rsync -avuzhP "${SERIAL_DIR}/" "${HEAD_EXP_DIR}/" # Copy output onto headnode

rm -rf "${SERIAL_DIR}"

# Delete all scratch home sub directories more than a week old.
find ${SCRATCH_HOME} -type d -name "*" -mtime +7 -printf "%T+ %p\n" | sort | cut -d ' ' -f 2- | sed -e 's/^/"/' -e 's/$/"/' | xargs rm -rf


echo "============"
echo "results synced"
