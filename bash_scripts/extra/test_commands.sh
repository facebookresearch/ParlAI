"""
Multitasking with MULTIWOZ 
"""
# from bart_submit_pft.py
# ./bart_prefinetune.sh \
#     -t wsc,wnli,wic,squad2,multiwoz_dst,google_sgd_dst,paraphrase_classification,wikisql,coqa \
#     -l 5e-6 -e 10 -n all_multitask \
#     -k 2021-11-18_00:34:03 \
#     -g $HOME/ParlAI/bash_scripts/slurm_logs/all_multitask_pft_lr_5e-6-2021-11-18_00:34:03.log \
#     -m "joint goal accuracy"

# from the line above
parlai multiprocessing_train \
    -m bart -eps 1 -bs 8 -opt adam -lr 5e-6 \
    -t "wsc,wnli,wic,squad2,multiwoz_dst:data_version=2.3,google_sgd_dst,paraphrase_classification,wikisql,coqa" \
    --eval-batchsize 16 \
    --evaltask multiwoz_dst \
    --fp16 true --warmup_updates 100 --warmup_rate 1e-5 \
    --log-every-n-secs 200 --validation-every-n-steps 50 --save-after-valid True \
    --validation-metric "multiwoz_dst/joint goal acc" --validation-metric-mode max --validation-patience 3 --validation_cutoff 100 --text-truncate 512 --label-truncate 512 --skip-generation False -tblog True \
    --model-file "models/bart_all_multitask_pft_test/model" \
    --max-train-steps 100 \
    --short_final_eval True --validation_max_exs 100 \
    --report-filename ${MF}.report.json \
    --world-logs ${MF}.world_logs.jsonl \


    # --evaltask "multiwoz_dst:data_version=2.3:val_reduced=True"

"""
Reverse SGD experiments (prefinetune with MultiWOZ included and finetune with SGD)
"""

# from bart_submit_pft.py
# ./bart_prefinetune.sh \
#     -t wsc,wnli,wic,squad2,multiwoz_dst:data_version=2.3,paraphrase_classification,wikisql,coqa \
#     -l 5e-6 -e 10 -n g_sgd_reversed \
#     -k 2021-11-18_01:07:56 \ 
#     -g $HOME/ParlAI/bash_scripts/slurm_logs/g_sgd_reversed_pft_lr_5e-6-2021-11-18_01:07:56.log \
#     -m loss -v False

# parlai multiprocessing_train \
#     -m bart -eps 1 -bs 8 -opt adam -lr 5e-6 \
#     -t "wsc,wnli,wic,squad2,multiwoz_dst:data_version=2.3,google_sgd_dst,paraphrase_classification,wikisql,coqa" \
#     --eval-batchsize 16 \
#     --fp16 true --warmup_updates 100 --warmup_rate 1e-5 \
#     --log-every-n-secs 200 --validation-every-n-steps 50 --save-after-valid True \
#     --validation-metric "multiwoz_dst/joint goal acc" --validation-metric-mode max --validation-patience 3 --validation_cutoff 0 --text-truncate 512 --label-truncate 512 --skip-generation False -tblog True \
#     --model-file "models/bart_all_multitask_pft_test/model" \
#     --max-train-steps 100 \
#     --short_final_eval True --validation_max_exs 100 \






