#!/bin/bash
#SBATCH --partition=hipri 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=16
#SBATCH --account=all
## %j is the job id, %u is the user id

# set -e 

source $HOME/miniconda/etc/profile.d/conda.sh
cd $HOME/CheckDST
source $HOME/CheckDST/set_envs_mine.sh
conda activate parlai_checkdst

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
VERSION=2.1
USEPROMPTS=False
LR=1e-4
FP16=True
JUST_TEST=False
MODEL_SIZE="t5-base"
DOMAIN="" # blank is all domains 
MFDIR="$HOME/CheckDST/ParlAI/models"
if [[ $DOMAIN == "" ]] ; then 
  MFDIR="$MFDIR/${MODEL_SIZE}/"
else 
  MFDIR="$MFDIR/${MODEL_SIZE}_${DOMAIN}/"
  ADD_OPT="$ADD_OPT --dom ${DOMAIN}" 
fi 

while getopts ":hs:l:i:m:p:v:f:b:g:u:t:" option; do
   case $option in
      h) # display Help
        Help
        exit;;
      m)
        MFDIR=$OPTARG;;
      i)
        ADD_OPT=$OPTARG;;
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
      u)
        UPDATE_FREQ=$OPTARG;;
      t) 
        JUST_TEST=$OPTARG;;
     \?) # Invalid option
        echo "Error: Invalid option"
        exit;;
   esac
done

MF="$MFDIR/model"
echo "Saving model to: ${MF}"
echo "Learning rate: ${LR}"
echo "seed: ${SD}"
echo "Version: ${VERSION}"
echo "Fewshot: ${FEWSHOT}"
echo "Use prompts: ${USEPROMPTS}"

if [[ $FEWSHOT == "True" ]]; then 
  N_EPOCH=20 
  SAVE_EPOCH=2
else 
  N_EPOCH=10
  SAVE_EPOCH=1
fi 

mkdir -p $MFDIR
CMD="\
parlai train_model \
    -m hugging_face/t5 --t5-model-arch $MODEL_SIZE \
    -t multiwoz_checkdst \
    --val_reduced False \
    --version $VERSION \
    --few_shot $FEWSHOT \
    --use_prompts $USEPROMPTS \
    --rand-seed $SD \
    --model-file $MF \
    -eps $N_EPOCH -bs $BATCH_SIZE --update-freq $UPDATE_FREQ  -opt adam -lr $LR \
    --fp16 True \
    --max_lr_steps 400000 \
    --max_train_time 144000 \
    --save-every-n-epochs $SAVE_EPOCH \
    --train_only False \
    --validation_every_n_epochs 1 --eval-batchsize 16 --validation-metric 'jga_original' --validation-metric-mode max \
    --optimize_for_multiwoz False \
    --skip-generation False \
    --warmup_updates 100 --warmup_rate 1e-5 \
    --log-every-n-secs 30 \
    --text-truncate 512 --label-truncate 512 \
    --dynamic-batching full \
    --lr-scheduler cosine \
    -tblog True \
    --report-filename ${MF}.report_fs_${FEWSHOT}.json \
    --world-logs ${MF}.world_logs_fs_${FEWSHOT}.jsonl \
    --just_test $JUST_TEST \
    $ADD_OPT |& tee ${MF}_log.txt
"


echo $CMD 
eval $CMD