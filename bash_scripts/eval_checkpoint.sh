#!/bin/bash
#SBATCH --job-name=eval_multiwoz2.2+
#SBATCH --partition=a100 
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/eval_checkpoint-%j.log

source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/ParlAI
conda activate parlai_internal

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#     --fp16 true \
#     -mf $1 \
#     --log-every-n-secs 100 \
#     --report-filename ".report" \
#     --data_version 2.3 \
#     --world-logs "${1}.world_logs.jsonl"

# echo $1 

# parlai multiprocessing_eval \
#     -dt test \
#     -m bart -t multiwoz_dst -bs 1 \
#     --fp16 true \
#     -mf $1 \
#     --log-every-n-secs 100 \
#     --report-filename ".report" \
#     --data_version 2.3 \
#     --world-logs "${1}.world_logs.jsonl" \
#     --test_reduced True

# echo $1 

parlai multiprocessing_eval \
    -dt test \
    -aug TP \
    -m bart -t multiwoz_dst_laug -bs 1 \
    --fp16 true \
    -mf $1 \
    --log-every-n-secs 100 \
    --report-filename ".report_TP" \
    --world-logs "${1}.world_logs_TP.jsonl" \
    --test_reduced True

echo $1 