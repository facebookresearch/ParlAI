# commands for finetuning on multiwoz and evaluating with checkdst 

MODEL=bart # 
# INIT_CONFIG="scratch" # one of [scratch, pft, muppet]
# INIT_CONFIG="pft" # one of [scratch, pft, muppet]
# INIT_CONFIG="muppet" # one of [scratch, pft, muppet]
INIT_CONFIG="soloist"
USE_SLURM=true # execute directly or submit as sbatch job

# for SEED in 0 ; do 
for SEED in 0 1 2 3 4 ; do 

    BATCH_SIZE=4
    UPDATE_FREQ=1
    FEWSHOT=False
    VERSION=2.3
    USEPROMPTS=True
    LR=5e-5
    FP16=True
    JUST_TEST=False

    MAINDIR="$PARLAI_DIR/models/${MODEL}_${INIT_CONFIG}_multiwoz${VERSION}"

    if [[ $INIT_CONFIG == "muppet" ]] ; then 
        INIT_CMD="--init-fairseq-model $PARLAI_DIR/data/models/bart_muppett/model.pt"
    elif [[ $INIT_CONFIG == "pft" ]] ; then 
        INIT_MODEL_PATH="$PARLAI_DIR/models/pre_emnlp/bart_all_pft/lr5e-6_eps10_ngpu8_bs8_2021-11-11_22:26:26/model"
        INIT_CMD="--init-model $INIT_MODEL_PATH"
    elif [[ $INIT_CONFIG == "soloist" ]] ; then 
        # update to path with best model 
        INIT_MODEL_PATH="$PARLAI_DIR/models/bart_+sgd_pft/lr5e-6_eps10_ngpu8_bs8_2022-06-24_02:54:40/model"
        INIT_CMD="--init-model $INIT_MODEL_PATH"
    else 
        INIT_MODEL_PATH="$PARLAI_DIR/data/models/bart/bart_large/model"
        INIT_CMD="--init-model $INIT_MODEL_PATH"
    fi 

    JOBNAME="${INIT_CONFIG}_sd${SEED}"
    if [[ $USE_SLURM == "true" ]] ; then 
        CMD_BEGIN="sbatch -J $JOBNAME"
    else 
        CMD_BEGIN="."
    fi 

    MFDIR="${MAINDIR}/fs_${FEWSHOT}_prompts_${USEPROMPTS}_lr${LR}_bs${BATCH_SIZE}_uf${UPDATE_FREQ}_sd${SEED}/"

    CMD="$CMD_BEGIN bart_multiwoz_finetune.sh \
        -m $MFDIR \
        -i \"$INIT_CMD\" \
        -s $SEED \
        -l $LR \
        -f $FEWSHOT \
        -p $USEPROMPTS \
        -v $VERSION \
        -b $BATCH_SIZE \
        -u $UPDATE_FREQ \
        -t $JUST_TEST \
        "
    echo $CMD
    eval $CMD
    # break 
done 


