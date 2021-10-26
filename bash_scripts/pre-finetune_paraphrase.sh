#!/bin/bash
#SBATCH --job-name=paraphrase_prefinetune
#SBATCH --partition=a100 
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/project/ParlAI/bash_scripts/slurm_logs/paraphrase_prefinetune-%j.log

source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/project/ParlAI
conda activate parlai

NOW=$(date +"%Y-%m-%d_%T")
BATCH_SIZE=8
MAX_TRAIN_STEPS="-1"
LR=1e-4

for sd in 3; do
    MF="models/gpt2_paraphrase/${NOW}_ngpu8_bs${BATCH_SIZE}_lr${LR}_eps20_sd${sd}/model"
    parlai multiprocessing_train \
        -m hugging_face/gpt2 -t paraphrase_classification -eps 20.3 -bs ${BATCH_SIZE} -opt adam -lr $LR \
        --eval-batchsize 1 \
        --fp16 true \
        --warmup_updates 100 \
        --warmup_rate 1e-5 \
        --log-every-n-secs 100 \
        --validation-every-n-epochs 1 \
        --save-after-valid True \
        --model-file $MF \
        --validation-metric 'f1' \
        --validation-metric-mode max \
        --add-special-tokens True \
        --validation-patience 5 \
        --rand-seed ${sd} \
        # --just-test True \
        # --val_reduced True

done