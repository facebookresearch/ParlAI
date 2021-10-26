#!/bin/bash
#SBATCH --job-name=pft_para_bart
#SBATCH --partition=a100 
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/bart_pft_para-%j.log


source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/ParlAI
conda activate parlai_internal

NOW=$(date +"%Y-%m-%d_%T")
BATCH_SIZE=8
MAX_TRAIN_STEPS="-1"
LR=1e-4
sd=0
# PFT_TASK="google_sgd_dst"
PFT_TASK="paraphrase_classification"

# for sd in 0; do
#     MF="models/bart_${PFT_TASK}/${NOW}_ngpu8_bs${BATCH_SIZE}_lr${LR}_sd${sd}/model"
#     parlai multiprocessing_train \
#         -m bart \
#         -t ${PFT_TASK} \
#         -eps 20 -bs $BATCH_SIZE -opt adam -lr $LR \
#         --eval-batchsize 1 \
#         --fp16 true \
#         --warmup_updates 100 \
#         --warmup_rate 1e-5 \
#         --log-every-n-secs 100 \
#         --validation-every-n-epochs 1 \
#         --save-after-valid True \
#         --model-file $MF \
#         --validation-metric 'joint goal acc' \
#         --validation-metric-mode max \
#         --validation-patience 8 \
#         --val_reduced True \
#         --text-truncate 512 \
#         --label-truncate 512 \
#         # --rand-seed ${sd} \
#         # --init-fairseq-model /data/home/justincho/ParlAI/data/models/bart_muppett/model.pt \
#         # --skip-generation True \

# done

for sd in 0; do
    MF="models/bart_${PFT_TASK}/${NOW}_ngpu8_bs${BATCH_SIZE}_lr${LR}_sd${sd}/model"
    parlai multiprocessing_train \
        -m bart \
        -t paraphrase_classification \
        -eps 20 -bs $BATCH_SIZE -opt adam -lr $LR \
        --eval-batchsize 1 \
        --fp16 true \
        --warmup_updates 100 \
        --warmup_rate 1e-5 \
        --log-every-n-secs 100 \
        --validation-every-n-epochs 1 \
        --save-after-valid True \
        --model-file $MF \
        --validation-metric 'loss' \
        --validation-metric-mode min \
        --validation-patience 8 \
        --val_reduced True \
        --text-truncate 512 \
        --label-truncate 512 \
        --skip-generation True \

done



