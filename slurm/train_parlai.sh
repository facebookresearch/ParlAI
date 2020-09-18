#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:2
#SBATCH --mem=32g  # Memory
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
export SERIAL_DIR="${SCRATCH_HOME}/${EXP_ID}"

export EXP_ID="${EXP_NAME}_${SLURM_JOB_ID}_${CURRENT_TIME}"
echo "Experiment ID: ${EXP_ID}"

cd "${EXP_ROOT}"

mkdir -p ${SERIAL_DIR}

echo "============"
echo "ParlAI Task========"

parlai train_model -m hugging_face/gpt2 --mf ${SERIAL_DIR}/${EXP_ID}.mod" \
 --add-special-tokens=True --add-start-token=True --gpt2-size=medium -t writing_prompts \
 --batchsize=4 --text-truncate=128 --label-truncate=128 --datapath="${DATASET_SOURCE}" \
 --save-after-valid=True --num-epochs=10 --max_train_time=3600 --validation-patience=3 \
 --validation-every-n-secs=10800 --validation-max-exs=5000

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
