#!/bin/bash

nvidia-smi

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh


echo
echo "======================================================================="
echo "Activating pytorch nightly"
echo "======================================================================="
conda deactivate
conda activate nightly
echo
echo "Collect env"
echo "------------------------------------------------------------"
python collect_env.py
echo

for mode in single dp ddp_single ddp_multi
do
    echo "Running mode=$mode"
    echo "------------------------------------------------------------"
    python -u memtestcase.py --mode=$mode 2>&1
    echo
done


