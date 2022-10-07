#!/bin/bash
#SBATCH --job-name=gpt_finetune
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id

source $HOME/miniconda/etc/profile.d/conda.sh
cd $HOME/ParlAI
conda activate parlai_internal

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Run evaluation script for a parlai model checkpoint"
   echo
   echo "Syntax: scriptTemplate [-h|h|m|i]"
   echo "options:"
   echo "f     Use few shot test set"
   echo "h     Print this Help."
   echo "m     pass model path"
   echo "i     init model path"
   echo "p     use prompts"
   echo "f     use fewshot setting"
   echo "v     specify multiwoz version"
   echo "b     batchsize"
   echo
}

############################################################
# Main Program                                             #
############################################################

# default values for testing with direct execution of this script instead of using python script: 
BATCH_SIZE=4
UPDATE_FREQ=1
SD=0
FEWSHOT=False
VERSION=2.3
USEPROMPTS=False
LR=1e-4
FP16=True
JUST_TEST=False


while getopts ":hs:l:i:m:p:v:f:b:g:u:t:" option; do
   case $option in
      h) # display Help
        Help
        exit;;
      m)
        MFDIR=$OPTARG;;
      i)
        INIT_CMD=$OPTARG;;
      s)
        SD=$OPTARG;;
      l) 
        LR=$OPTARG;;
      f) 
        FEWSHOT=$OPTARG;;
      p) 
        USEPROMPTS=$OPTARG;;
      v)
        VERSION=$OPTARG;;
      b)
        BATCH_SIZE=$OPTARG;;
      g)
        LOGFILE=$OPTARG;;
      u)
        UPDATE_FREQ=$OPTARG;;
      t) 
        JUST_TEST=$OPTARG;;
     \?) # Invalid option
        echo "Error: Invalid option"
        exit;;
   esac
done

MF="${MFDIR}model"
echo "Learning rate: ${LR}"
echo "seed: ${SD}"
echo "Version: ${VERSION}"
echo "Fewshot: ${FEWSHOT}"
echo "Use prompts: ${USEPROMPTS}"

if [ $FEWSHOT = "True" ]; then 
  N_EPOCH=20 
  SAVE_EPOCH=2
else 
  N_EPOCH=10
  SAVE_EPOCH=1
fi 

# MF="models/gpt2_${DOWNSTREAM}_ft_multiwoz${VERSION}/${NOW}_ngpu1_bs${BATCH_SIZE}_lr${LR}_eps20_fewshot_${FEWSHOT}_sd${sd}/model"
# INIT_MODEL="$HOME/ParlAI/models/gpt2_sgd_dst/2021-10-03_19:21:58_bs4_lr5e-4_eps20_sd0/model"
# INIT_MODEL="$HOME/ParlAI/models/gpt2_paraphrase/2021-10-11_18:56:52_ngpu8_bs8_lr1e-4_eps20_sd0/model"

CMD="\
parlai train_model \
    -m hugging_face/gpt2 \
    -t multiwoz_dst \
    --val_reduced_size -1 \
    --version $VERSION \
    --few_shot $FEWSHOT \
    --use_prompts $USEPROMPTS \
    --rand-seed ${SD} \
    --model-file $MF \
    -eps $N_EPOCH -bs $BATCH_SIZE --update-freq $UPDATE_FREQ -opt adam -lr $LR \
    --fp16 $FP16 \
    --max_lr_steps 400000 \
    --max_train_time 144000 \
    --save-every-n-epochs $SAVE_EPOCH \
    --train_only True \
    --optimize_for_multiwoz True \
    --skip-generation True \
    --warmup_updates 100 --warmup_rate 1e-5 \
    --log-every-n-secs 30 \
    --add-special-tokens True \
    --dynamic-batching full \
    --lr-scheduler cosine \
    -tblog True \
    --report-filename ${MF}.report_fs_${FEWSHOT}.json \
    --world-logs ${MF}.world_logs_fs_${FEWSHOT}.jsonl 
    --just_test $JUST_TEST
    $INIT_CMD
"

echo $CMD 
eval $CMD

echo "Copy slurm log file into the model directory"
cp $LOGFILE $MFDIR