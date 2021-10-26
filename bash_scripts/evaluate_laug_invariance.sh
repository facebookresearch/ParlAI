#!/bin/bash
#SBATCH --job-name=eval_laug_inv
#SBATCH --partition=a100 
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/eval_laug_inv-%j.log


source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/ParlAI
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
   echo "i     Type of invariance to use. Should be one of SD, TP, NEI"
   echo
}

############################################################
# Main Program                                             #
############################################################


while getopts ":hm:i:f:d:" option; do
   case $option in
      f)
        FEWSHOT=$OPTARG;;
      h) # display Help
        Help
        exit;;
      i)
        INVARIANCE=$OPTARG;;
      d)
        MODEL=$OPTARG;;
      m)
        MF=$OPTARG
        if [[ "$MF" != *"model" ]]; then 
            echo $MF
            echo "Error: need to have 'model' as part of the path"
            exit
        fi;;
     \?) # Invalid option
        echo "Error: Invalid option"
        exit;;
   esac
done

echo $MF
echo $FEWSHOT
echo $INVARIANCE

# parlai multiprocessing_eval \
parlai eval_model \
    -m $MODEL \
    --model-file $MF \
    --datatype test \
    --batchsize 1 \
    --task multiwoz_dst_laug \
    --report-filename ".${INVARIANCE}_report_fs_${FEWSHOT}.json" \
    --world-logs "${MF}.${INVARIANCE}_world_logs_fs_${FEWSHOT}.jsonl" \
    -aug $INVARIANCE \
    -fs $FEWSHOT \

# looking at the data 

# parlai dd --task multiwoz_dst_laug -dt test -aug TP