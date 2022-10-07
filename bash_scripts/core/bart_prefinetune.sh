#!/bin/bash
#SBATCH --partition=a100 
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Run prefinetuning script for a parlai model"
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


while getopts ":hs:l:t:e:n:k:m:v:" option; do
   case $option in
      h) # display Help/
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
VALSETTINGS="--validation-metric loss --validation-metric-mode min --validation_cutoff 0 "
# VALSETTINGS="--validation-metric 'joint goal acc' --validation-metric-mode max --validation_cutoff 100"
VALSETTINGS="$VALSETTINGS --skip-generation True"

MFDIR="models/bart_${TASK_ALIAS}_pft/lr${LR}_eps${EPOCHS}_ngpu8_bs${BATCH_SIZE}_${NOW}/"
MF="${MFDIR}model"
CMD="\
parlai multiprocessing_train \
    -m bart \
    -eps $EPOCHS -bs $BATCH_SIZE -opt adam -lr $LR \
    -t $TASK \
    --eval-batchsize 32 \
    --fp16 true \
    --warmup_updates 100 \
    --warmup_rate 1e-5 \
    --log-every-n-secs 200 \
    --validation-every-n-steps 2500 \
    --save-after-valid True \
    --model-file $MF \
    --validation-patience 3 \
    --text-truncate 512 \
    --label-truncate 512 \
    -tblog True \
    --report-filename ${MF}.report.json \
    --world-logs ${MF}.world_logs.jsonl \
    $VALSETTINGS \
    $EVALTASK |& tee ${MFDIR}log.txt
"

echo $CMD
eval $CMD

# # whether to run evaluation after completing training
# if [[ "$RUN_EVAL" = True && -e "$MF.test" ]]; then 
#   for INV in SD TP NEI; do 
#     for FEWSHOT in True False; do
#       sbatch evaluate_laug_invariance.sh -d bart -i $INV -m $MF -f $FEWSHOT
#     done
#   done
# else
#   echo "Model training was not completed for this directory. Cannot find: $MF.test"
# fi