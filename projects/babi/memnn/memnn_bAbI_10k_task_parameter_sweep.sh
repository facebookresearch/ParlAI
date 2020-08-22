#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Reproduce results from https://arxiv.org/abs/1503.08895 bAbI tasks 1-3 with
# 10k training examples

PARLAI=../../..                # Where ParlAI exists in filesystem relative to script

DATE=`date +"%Y%m%d"`
SWEEP_NAME=memnn_bAbI_task10k_t1to3_parameter_sweep  # Sweep name
JOBSCRIPTS=scripts
mkdir -p ${JOBSCRIPTS}

SAVE_ROOT=/tmp/${DATE}/${SWEEP_NAME}    # Where to save the model files
mkdir -p stdout stderr

# Training parameters
ltim=10                                # log every 10 seconds
vtim=900                               # run validation every 900 seconds
vme=10000                              # max examples for validation
vp=-1                                  # validation patience; -1 means infinite patience
vmt=f1                                 # validation metric
ttim=252000                            # max training time (=70 hours)
bs=128                                 # batch size
dmf=20                                 # min frequency of words to keep in dictionary
dmexs=10000000                         # max number of examples to build dict on


# Job Creation Loop(s)
for lr in 0.01 0.001; do               # learning rate
for esz in 32 64 128; do               # embedding size
for hops in 2 3; do                    # memory hops
for memsize in 100; do                 # memory size
for tfs in 1; do                       # whether to use time features
for penc in 0 1; do                    # whether to use positional encoding
for dr in 0.1; do                      # dropout
for optm in adam; do                   # optimizer
for output in rank; do                 # output type
for task in 1 2 3; do                  # bAbI tasks to do
        PARAMS=lr-${lr}_esz-${esz}_hps-${hops}_memsize-${memsize}_tfs-${tfs}\
_penc-${penc}_dr-${dr}_output-${output}_optm-${optm}_task-${task}
        SAVE=${SAVE_ROOT}.${PARAMS}
        mkdir -p ${SAVE}
        JNAME=${SWEEP_NAME}.${PARAMS}
        SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
        SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm
        echo "#!/bin/sh" > ${SCRIPT}
        echo "#!/bin/sh" > ${SLURM}
        echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
        echo "#SBATCH --output=stdout/${JNAME}.%j" >> ${SLURM}
        echo "#SBATCH --error=stderr/${JNAME}.%j" >> ${SLURM}
        echo "#SBATCH --partition=learnfair" >> ${SLURM}
        echo "#SBATCH --nodes=1" >> ${SLURM}
        echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
        echo "#SBATCH --time=72:00:00" >> ${SLURM}
        echo "#SBATCH --signal=USR1" >> ${SLURM}
        echo "#SBATCH -c 2" >> ${SLURM}
        echo "#SBATCH --gres=gpu:1" >> ${SLURM}
        echo "srun sh ${SCRIPT}" >> ${SLURM}
        echo "echo \$SLURM_JOB_ID >> jobs" >> ${SCRIPT}
        echo "{ " >> ${SCRIPT}
        echo "echo $SWEEP_NAME $BSZ " >> ${SCRIPT}
        echo "nvidia-smi" >> ${SCRIPT}
        echo "cd $PARLAI" >> ${SCRIPT}
        echo parlai train_model \
            -t babi:task10k:${task} \
            -ltim ${ltim} -vtim ${vtim} -vme ${vme} -vp ${vp} -vmt ${vmt} -ttim ${ttim} \
            -bs ${bs} -lr ${lr} --embedding-size ${esz} --hops ${hops} \
            --mem-size ${memsize} --time-features ${tfs} --dropout ${dr} \
            --position-encoding ${penc} --output ${output} --optimizer ${optm} \
            -m memnn -mf ${SAVE}/model \
            --dict-minfreq ${dmf} --dict-maxexs ${dmexs} >> ${SCRIPT}
        echo "nvidia-smi" >> ${SCRIPT}
        echo "kill -9 \$\$" >> ${SCRIPT}
        echo "} & " >> ${SCRIPT}
        echo "child_pid=\$!" >> ${SCRIPT}
        echo "trap \"echo 'Signal received'; kill -9 \$child_pid; sbatch ${SLURM}; exit 0;\" USR1" >> ${SCRIPT}
        echo "while true; do     sleep 1; done" >> ${SCRIPT}
        sbatch ${SLURM}
done
done
done
done
done
done
done
done
done
done
