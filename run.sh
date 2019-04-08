#!/bin/bash

nvidia-smi

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


