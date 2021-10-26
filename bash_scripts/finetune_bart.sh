#!/bin/bash
#SBATCH --job-name=bart_muppet_scratch_multiwoz2.3
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/bart_multiwoz2.3-%j.log


source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/ParlAI
conda activate parlai_internal

NOW=$(date +"%Y-%m-%d_%T")
BATCH_SIZE=4
MAX_TRAIN_STEPS="-1"
LR=1e-4
MAX_TRAIN_TIME=82800
sd=0
REDUCE_FACTOR=1
VERSION=2.3
FEWSHOT=False
USEPROMPTS=True

# from Kun's run.sh

for sd in 7; do
    MF="models/bart_muppet_multiwoz${VERSION}/${NOW}_ngpu1_bs${BATCH_SIZE}_lr${LR}_eps20_fewshot_${FEWSHOT}_prompts_${USEPROMPTS}_sd${sd}/model"
    parlai train_model \
        -m bart \
        -t multiwoz_dst \
        -eps 20 -bs $BATCH_SIZE -opt adam -lr $LR \
        --eval-batchsize 1 \
        --fp16 true \
        --warmup_updates 100 \
        --warmup_rate 1e-5 \
        --log-every-n-secs 100 \
        --validation-every-n-epochs 1 \
        --save-after-valid True \
        --model-file $MF \
        --validation-metric 'joint goal acc' \
        --validation-metric-mode max \
        --validation-patience 8 \
        --rand-seed ${sd} \
        --data_version $VERSION \
        --val_reduced True \
        --reduce_train_factor $REDUCE_FACTOR \
        --few_shot $FEWSHOT \
        --text-truncate 512 \
        --label-truncate 512 \
        --use_prompts True \
        --init-fairseq-model /data/home/justincho/ParlAI/data/models/bart_muppett/model.pt \
        # --special-tok-lst "<user>,<system>"
        # --skip-generation True \



done





