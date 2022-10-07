

NGPU=8
for LR in "1e-5" "5e-5" "5e-6"; do
    for SD in 0 1 2; do 
        for BATCH_SIZE in 8; do 
            for UPDATE_FREQ in 1; do 
                # from scratch 
                MF="models/bart_scratch_sgd_ft/lr${LR}_bs${BATCH_SIZE}_uf${UPDATE_FREQ}_npu${NGPU}_sd${SD}_${NOW}/model"
                INIT_CMD="--init-model $PARLAI_DIR/data/models/bart/bart_large/model"

                LOG_DIR="$PARLAI_DIR/bash_scripts/slurm_logs/sgd_scratch_ft_lr${LR}_bs_${BATCH_SIZE}_uf_${UPDATE_FREQ}_npu${NGPU}/"
                if [[ ! -e $LOG_DIR ]]; then 
                    mkdir -p $LOG_DIR
                fi 

                LOG_FILE="${LOG_DIR}${NOW}.log"
                NOW=$(date +"%Y-%m-%d_%T")
                sleep 1 

                # --gres=gpu:${NGPU} --time=24:00:00

                CMD="sbatch --job-name sgd_scratch_ft --output=${LOG_FILE} --gres=gpu:${NGPU} --time=24:00:00 \
                    bart_finetune_sgd.sh  \
                    -m $MF \
                    -i \"$INIT_CMD\" \
                    -s $SD \
                    -l $LR \
                    -b $BATCH_SIZE \
                    -u $UPDATE_FREQ \
                    -g $LOG_FILE \
                "
                echo $CMD
                eval $CMD

                NOW=$(date +"%Y-%m-%d_%T")
                sleep 1 
                # SGD finetuning commands (from PFT model)
                MF="models/bart_pft_sgd_ft/lr${LR}_bs${BATCH_SIZE}_uf${UPDATE_FREQ}_npu${NGPU}_sd${SD}_${NOW}/model"
                INIT_CMD="--init-model $PARLAI_DIR/models/bart_g_sgd_reversed_pft/lr5e-6_eps10_ngpu8_bs8_2021-11-15_11:45:04/model"

                LOG_DIR="$PARLAI_DIR/bash_scripts/slurm_logs/sgd_pft_ft_lr${LR}_bs_${BATCH_SIZE}_uf_${UPDATE_FREQ}_npu${NGPU}/"
                if [[ ! -e $LOG_DIR ]]; then 
                    mkdir -p $LOG_DIR
                fi 
                LOG_FILE="${LOG_DIR}${NOW}.log"

                CMD="sbatch --job-name sgd_pft_ft --output=${LOG_FILE} --gres=gpu:${NGPU} --time=24:00:00 \
                    bart_finetune_sgd.sh  \
                    -m $MF \
                    -i \"$INIT_CMD\" \
                    -s $SD \
                    -l $LR \
                    -b $BATCH_SIZE \
                    -u $UPDATE_FREQ \
                    -g $LOG_FILE \
                "

                echo $CMD
                eval $CMD
            done
        done
    done
done

# parlai train_model -m bart -t google_sgd_dst --val_reduced True --use_prompts True --rand-seed 2 --model-file models/bart_scratch_sgd_ft/lr5e-6_bs8_uf1_sd2_2021-11-18_05:39:03/model -eps 20 -bs 8 --update-freq 1 -opt adam -lr 5e-6 --fp16 true --max_lr_steps 40000 --max_train_time 14400 --warmup_updates 100 --warmup_rate 1e-5 --log-every-n-secs 20 --validation-every-n-epochs 1 --save-after-valid True --eval-batchsize 1 --validation-metric 'joint goal acc' --validation-metric-mode max --validation-patience 5 --text-truncate 512 --label-truncate 512  --lr-scheduler cosine -tblog True --report-filename models/bart_scratch_sgd_ft/lr5e-6_bs8_uf1_sd2_2021-11-18_05:39:03/model.report.json --world-logs models/bart_scratch_sgd_ft/lr5e-6_bs8_uf1_sd2_2021-11-18_05:39:03/model.world_logs.jsonl --init-model $PARLAI_DIR/data/models/bart/bart_large/model --just_test True