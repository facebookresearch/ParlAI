#!/bin/bash
#SBATCH --partition=a100 
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id



source $HOME/miniconda/etc/profile.d/conda.sh
cd $HOME/CheckDST/ParlAI
conda activate parlai_internal

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Run evaluation script for a parlai model checkpoint"
   echo
   echo "Syntax: scriptTemplate [-h|l|t|s|]"
   echo "options:"
   echo "l     learning rate"
   echo "h     Print this Help."
   echo "t     tasks"
   echo "s     random seed"
   echo "e     epochs"
   echo "n     alias to use for saving directory name"
   echo
}

############################################################
# Main Program                                             #
############################################################


while getopts ":hs:l:t:e:n:k:g:m:v:" option; do
   case $option in
      h) # display Help
        Help
        exit;;
      l)
        LR=$OPTARG;;
      t)
        TASK=$OPTARG;;
      s)
        SD=$OPTARG;;
      e) 
        EPOCHS=$OPTARG;;
      n) 
        TASK_ALIAS=$OPTARG;; 
      k) 
        NOW=$OPTARG;;
      g) 
        LOGFILE=$OPTARG;;
      m) 
        VALSETTINGS=$OPTARG;; 
      v)
        RUN_EVAL=$OPTARG;; 
     \?) # Invalid option
        echo "Error: Invalid option"
        exit;;
   esac
done

echo "Learning rate: ${LR}"
echo "Tasks: ${TASK}"
echo "Task alias: ${TASK_ALIAS}"
echo "Epochs: ${EPOCHS}"
echo "val settings: $VALSETTINGS"

BATCH_SIZE=8
# EVALTASK="--evaltask multiwoz_dst:version=2.3"
EVALTASK=""
# VALSETTINGS="--validation-metric loss --validation-metric-mode min --validation_cutoff 0"
VALSETTINGS="--validation-metric 'joint goal acc' --validation-metric-mode max --validation_cutoff 100"

MFDIR="models/gpt2_${TASK_ALIAS}_pft/lr${LR}_eps${EPOCHS}_ngpu8_bs${BATCH_SIZE}_${NOW}/"
MF="${MFDIR}model"
CMD="\
parlai multiprocessing_train \
    -m hugging_face/gpt2 \
    -eps $EPOCHS -bs $BATCH_SIZE -opt adam -lr $LR \
    -t $TASK \
    --eval-batchsize 16 \
    --fp16 true \
    --warmup_updates 100 \
    --warmup_rate 1e-5 \
    --log-every-n-secs 200 \
    --max-train-steps 40000 \
    --max-train-time 36000 \
    --validation-every-n-epochs 1 \
    --save-after-valid True \
    --model-file $MF \
    --validation-patience 3 \
    --skip-generation False \
    --add-special-tokens True \
    -tblog True \
    --dynamic_batching full \
    --lr-scheduler cosine \
    --val-reduced True \
    --test-reduced True \
    --report-filename ${MF}.report.json \
    --world-logs ${MF}.world_logs.jsonl \
    $VALSETTINGS \
    $EVALTASK
"

echo $CMD
eval $CMD

# copy slurm log file into model directory 
if [[ -e $LOGFILE && -e $MFDIR ]]; then 
  cp $LOGFILE $MFDIR
else
  if [[ ! -e $LOGFILE ]]; then 
    echo "Could not find the log file: $LOGFILE"
  elif [[ ! -e $MFDIR ]]; then 
    echo "Could not find the model dir: $MFDIR"
  fi 
fi 

