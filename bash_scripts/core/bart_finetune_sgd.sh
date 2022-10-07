#!/bin/bash
#SBATCH --partition=a100 
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10


source $HOME/miniconda/etc/profile.d/conda.sh
cd $HOME/CheckDST/ParlAI
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
VERSION=2.3
USEPROMPTS=True
LR=1e-4
TRAIN_CMD="train_model"
FP16=true

while getopts ":hs:l:i:m:p:v:f:b:g:u:" option; do
   case $option in
      h) # display Help
        Help
        exit;;
      m)
        MF=$OPTARG;;
      i)
        INIT_CMD=$OPTARG;;
      s)
        SD=$OPTARG;;
      l) 
        LR=$OPTARG;;
      p) 
        USEPROMPTS=$OPTARG;;
      b)
        BATCH_SIZE=$OPTARG;;
      g)
        LOGFILE=$OPTARG;;
      u)
        UPDATE_FREQ=$OPTARG;;
     \?) # Invalid option
        echo "Error: Invalid option"
        exit;;
   esac
done

echo "Learning rate: ${LR}"
echo "seed: ${SD}"

# TRAIN_CMD="train_model"
TRAIN_CMD="multiprocessing_train"

CMD="\
parlai $TRAIN_CMD \
    -m bart \
    -t google_sgd_dst \
    --val_reduced True \
    --test_reduced True \
    --use_prompts $USEPROMPTS \
    --rand-seed $SD \
    --model-file $MF \
    -eps 20 -bs $BATCH_SIZE --update-freq $UPDATE_FREQ  -opt adam -lr $LR \
    --fp16 $FP16 \
    --max_lr_steps 40000 \
    --max_train_time 36000 \
    --warmup_updates 100 --warmup_rate 1e-5 \
    --log-every-n-secs 20 \
    --validation-every-n-epochs 1 --save-after-valid True --eval-batchsize 16 --validation-metric 'joint goal acc' --validation-metric-mode max --validation-patience 5 \
    --text-truncate 512 --label-truncate 512 \
    --dynamic-batching full \
    --lr-scheduler cosine \
    -tblog True \
    --report-filename ${MF}.report.json \
    --world-logs ${MF}.world_logs.jsonl \
    $INIT_CMD \
"


echo $CMD 
eval $CMD

echo "Copy slurm log file into the model directory"
cp $LOGFILE $MFDIR