#!/bin/bash
#SBATCH --job-name=gpt2_scratch_multiwoz2.3
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/gpt2_scratch_multiwoz2.3-%j.log

source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/ParlAI
conda activate parlai_internal

# source /data/home/justincho/ParlAI/bash_scripts/base.sh

NOW=$(date +"%Y-%m-%d_%T")
BATCH_SIZE=8
MAX_TRAIN_STEPS="-1"
LR=5e-4
MAX_TRAIN_TIME=82800
sd=0
REDUCE_FACTOR=1
VERSION=2.3
FEWSHOT=False

# from Kun's run.sh

for sd in 0; do
    # MF="models/gpt2_scratch_multiwoz${VERSION}/${NOW}_ngpu1_bs${BATCH_SIZE}_lr${LR}_eps20_fewshot_${FEWSHOT}_sd${sd}/model"
    # parlai multiprocessing_train \
    #     -m hugging_face/gpt2 \
    #     -t multiwoz_dst \
    #     -eps 20 -bs $BATCH_SIZE -opt adam -lr $LR \
    #     --eval-batchsize 1 \
    #     --fp16 true \
    #     --warmup_updates 100 \
    #     --warmup_rate 1e-5 \
    #     --log-every-n-secs 100 \
    #     --validation-every-n-epochs 1 \
    #     --save-after-valid True \
    #     --model-file $MF \
    #     --validation-metric 'joint goal acc' \
    #     --validation-metric-mode max \
    #     --add-special-tokens True \
    #     --validation-patience 5 \
    #     --rand-seed ${sd} \
    #     --data_version $VERSION \
    #     --val_reduced True \
    #     --reduce_train_factor $REDUCE_FACTOR \
    #     --few_shot $FEWSHOT \
        # --skip-generation True \


#     mkdir -p experiment/gen_gpt2_nodict_sd${sd}/

#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#         --fp16 true \
#         -mf $MF \
#         --log-every-n-secs 100 \
#         --report-filename experiment/gen_gpt2_nodict_sd${sd}/model.report \
#         --world-logs experiment/gen_gpt2_nodict_sd${sd}/model.worldlogs.jsonl

#     cd experiment/gen_gpt2_nodict_sd${sd}/
#     cp model.worldlogs.jsonl result_test.jsonl
#     cd ../../

    AUG=SD
    MF="models/gpt2_scratch_LAUG_${AUG}_multiwoz${VERSION}/${NOW}_ngpu1_bs${BATCH_SIZE}_lr${LR}_eps20_fewshot_${FEWSHOT}_sd${sd}/model"
    parlai multiprocessing_train \
        -m hugging_face/gpt2 \
        -t multiwoz_dst_laug \
        --evaltask multiwoz_dst \
        -aug $AUG \
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
        --add-special-tokens True \
        --validation-patience 5 \
        --rand-seed ${sd} \
        --data_version $VERSION \
        --val_reduced True \
        --reduce_train_factor $REDUCE_FACTOR \
        --few_shot $FEWSHOT \


done

# bart
# for sd in 0; do
#     model="bart"
#     LR=1e-4
#     MF="models/${model}_scratch_multiwoz${VERSION}/${NOW}_ngpu1_bs${BATCH_SIZE}_lr${LR}_eps20_reduced${REDUCE_FACTOR}_sd${sd}/model"
#     parlai train_model \
#         -m bart -t multiwoz_dst -eps 20 -bs $BATCH_SIZE -opt adam -lr $LR \
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
#         --validation-patience 5 \
#         --rand-seed ${sd} \
#         --data_version $VERSION \
#         --val_reduced True \
#         --reduce_train_factor $REDUCE_FACTOR \
#         --text-truncate 512 --label-truncate 512 \
#         --skip-generation True \
#         # --dict-class parlai.agents.hugging_face.dict:Gpt2DictionaryAgent \
#         # --dict-tokenizer re \

# done



