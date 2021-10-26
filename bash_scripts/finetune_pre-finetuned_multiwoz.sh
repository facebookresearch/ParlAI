#!/bin/bash
#SBATCH --job-name=para_finetune_multiwoz2.2
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/sgd_finetune_multiwoz2.2-%j.log

source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/ParlAI
conda activate parlai_internal

NOW=$(date +"%Y-%m-%d_%T")
BATCH_SIZE=8
MAX_TRAIN_STEPS="-1"
LR=5e-4
MAX_TRAIN_TIME=82800
REDUCE_FACTOR=1
VERSION=2.3
# FEWSHOT=True
FEWSHOT=False
DOWNSTREAM=para
# DOWNSTREAM=sgd

for sd in 1; do
    MF="models/gpt2_${DOWNSTREAM}_ft_multiwoz${VERSION}/${NOW}_ngpu1_bs${BATCH_SIZE}_lr${LR}_eps20_fewshot_${FEWSHOT}_sd${sd}/model"
    # INIT_MODEL="/data/home/justincho/ParlAI/models/gpt2_sgd_dst/2021-10-03_19:21:58_bs4_lr5e-4_eps20_sd0/model"
    INIT_MODEL="/data/home/justincho/ParlAI/models/gpt2_paraphrase/2021-10-11_18:56:52_ngpu8_bs8_lr1e-4_eps20_sd0/model"
    parlai multiprocessing_train \
        -m hugging_face/gpt2 -t multiwoz_dst -eps 20 -bs $BATCH_SIZE -opt adam -lr $LR \
        --eval-batchsize 1 \
        --fp16 true \
        --warmup_updates 100 \
        --warmup_rate 1e-5 \
        --log-every-n-secs 100 \
        --validation-every-n-epochs 1 \
        --save-after-valid True \
        --model-file $MF \
        --init_model $INIT_MODEL \
        --validation-metric 'joint goal acc' \
        --validation-metric-mode max \
        --add-special-tokens True \
        --validation-patience 5 \
        --rand-seed ${sd} \
        --data_version ${VERSION} \
        --val_reduced True \
        --reduce_train_factor $REDUCE_FACTOR \
        # --few_shot $FEWSHOT \
        # --skip-generation True \


    # todo add script for running all evaluations for other metrics, or add them to multiwoz_dst task using flags
done


# parlai train_model \
# parlai multiprocessing_train \
#     -m bart \
#     -mf "models/bart_finetuned_multiwoz2.2+/${NOW}_${BATCH_SIZE}_${MAX_TRAIN_STEPS}_${LR}/model" \
#     --init_model $BASE_MODEL \
#     -t multiwoz_dst \
#     --data_version 2.2+ \
#     -bs $BATCH_SIZE \
#     --update-freq 8 \
#     --max-train-steps $MAX_TRAIN_STEPS\
#     --max-train-time $MAX_TRAIN_TIME\
#     --optimizer adam \
#     --gradient-clip 0.1\
#     --learningrate $LR  \
#     --warmup_updates 1000 \
#     --lr-scheduler reduceonplateau \
#     --lr-scheduler-decay 0.5 \
#     --lr-scheduler-patience 3 \
#     --text-truncate 512 --label-truncate 512 \
#     --log_every_n_steps 30 \
#     --history-size 15 \
#     --fp16 true --fp16-impl mem_efficient \
#     --num-workers 8 \
#     --eval-batchsize 16 --validation-metric "joint goal acc" --validation-metric-mode max --validation_cutoff 100 --validation-every-n-steps 2000 -sval True \
#     --validation-patience 8 \
    # --skip-generation True 
    # --dynamic-batching full \
