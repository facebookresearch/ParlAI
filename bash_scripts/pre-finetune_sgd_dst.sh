#!/bin/bash
#SBATCH --job-name=sgd_dst_prefinetune
#SBATCH --partition=a100 
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/project/ParlAI/bash_scripts/slurm_logs/sgd_dst_prefinetune-%j.log

source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/project/ParlAI
conda activate parlai

NOW=$(date +"%Y-%m-%d_%T")
BATCH_SIZE=4
MAX_TRAIN_STEPS="-1"
LR=1e-5
MAX_TRAIN_TIME=86400

# googel_sgd_dst format is sort of incorrect. this can be circumvented by simply setting history to 1 
# parlai multiprocessing_train \
#     -m bart \
#     -mf "models/bart_sgd_dst_prefinetune/${NOW}_${BATCH_SIZE}_${MAX_TRAIN_STEPS}_${LR}/model" \
#     -t google_sgd_dst\
#     -bs $BATCH_SIZE \
#     --update-freq 8 \
#     --max-train-steps $MAX_TRAIN_STEPS\
#     --max-train-time $MAX_TRAIN_TIME\
#     --optimizer adam \
#     --learningrate $LR  \
#     --warmup_updates 1000 \
#     --lr-scheduler reduceonplateau \
#     --lr-scheduler-decay 0.5 \
#     --lr-scheduler-patience 3 \
#     --text-truncate 768 --label-truncate 256 \
#     --log_every_n_steps 30 \
#     --history-size 1 \
#     --fp16 true --fp16-impl mem_efficient \
#     --eval-batchsize 16 --validation-metric "joint goal acc" --validation-metric-mode max --validation_cutoff 100 --validation-every-n-steps 1500 -sval True \
#     --val_reduced True \
#     --num-workers 8 \
#     --validation-patience 8 \
    # --init-model /data/home/justincho/project/ParlAI/models/bart_sgd_dst_prefinetune/2021-09-25_08:57:25_4_-1_1e-5/model\
    # --just_test True \
    # --skip-generation True
    # --dynamic-batching full \
    # --eval-batchsize 128 --validation-metric loss --validation-metric-mode min --validation-every-n-steps 2500 -sval True \


# parlai multiprocessing_train \
#     -m bart \
#     -mf "models/bart_sgd_dst_prefinetune/${NOW}_${BATCH_SIZE}_${MAX_TRAIN_STEPS}_${LR}/model" \
#     -t google_sgd_dst\
#     -bs $BATCH_SIZE \
#     --update-freq 8 \
#     --max-train-steps $MAX_TRAIN_STEPS\
#     --max-train-time $MAX_TRAIN_TIME\
#     --optimizer adam \
#     --learningrate $LR  \
#     --warmup_updates 1000 \
#     --lr-scheduler reduceonplateau \
#     --lr-scheduler-decay 0.5 \
#     --lr-scheduler-patience 3 \
#     --text-truncate 768 --label-truncate 256 \
#     --log_every_n_steps 30 \
#     --history-size 1 \
#     --fp16 true --fp16-impl mem_efficient \
#     --eval-batchsize 16 --validation-metric "joint goal acc" --validation-metric-mode max --validation_cutoff 100 --validation-every-n-steps 1500 -sval True \
#     --val_reduced True \
#     --num-workers 8 \
#     --validation-patience 8 \


for sd in 0; do
    MF="models/gpt2_sgd_dst/${NOW}_bs4_lr5e-4_eps20_sd${sd}/model"
    parlai multiprocessing_train \
        -m hugging_face/gpt2 -t google_sgd_dst -eps 20.3 -bs 8 -opt adam -lr 5e-4 \
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
        --add-special-tokens True \
        --validation-patience 5 \
        --val_reduced True
        # --rand-seed ${sd} \


    mkdir -p experiment/sgd_dst_gen_gpt2_nodict_sd${sd}/

    parlai multiprocessing_eval \
        -dt test \
        -m hugging_face/gpt2 -t google_sgd_dst -bs 1 \
        --fp16 true \
        -mf $MF \
        --log-every-n-secs 100 \
        --report-filename experiment/sgd_dst_gen_gpt2_nodict_sd${sd}/model.report \
        --world-logs experiment/sgd_dst_gen_gpt2_nodict_sd${sd}/model.worlds.jsonl

    cd experiment/sgd_dst_gen_gpt2_nodict_sd${sd}/
    cp model.worlds.jsonl result_test.jsonl
    cd ../../
done