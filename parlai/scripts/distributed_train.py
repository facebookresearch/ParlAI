#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed training script. NOT MEANT TO BE CALLED DIRECTLY BY USER.

This script is meant to be in conjunction with
`SLURM <https://slurm.schedmd.com/>`, which provides environmental variables
describing the environment.

An example sbatch script is below, for a 2-host, 8-GPU setup (16 total gpus):

.. code-block:: bash\n\n

  #!/bin/sh
  #SBATCH --job-name=distributed_example
  #SBATCH --output=/path/to/savepoint/stdout.%j
  #SBATCH --error=/path/to/savepoint/stderr.%j
  #SBATCH --partition=priority
  #SBATCH --nodes=2
  #SBATCH --time=0:10:00
  #SBATCH --signal=SIGINT
  #SBATCH --gres=gpu:8
  #SBATCH --ntasks-per-node=8
  #SBATCH --mem=64G
  #SBATCH --cpus-per-task=10
  srun python -u -m parlai.scripts.distributed_train \
    -m seq2seq -t convai2 --dict-file /path/to/dict-file

"""

import os
import socket
import subprocess

import parlai.scripts.train_model as single_train
from parlai.scripts.multiprocessing_train import multiprocess_train


def main():
    # double check we're using SLURM
    node_list = os.environ.get('SLURM_JOB_NODELIST')
    if node_list is None:
        raise RuntimeError(
            'Does not appear to be in a SLURM environment. '
            'You should not call this script directly; see launch_distributed.py'
        )

    parser = single_train.setup_args()
    parser.add_distributed_training_args()
    parser.add_argument('--port', type=int, default=61337, help='TCP port number')
    opt = parser.parse_args(print_args=(os.environ['SLURM_PROCID'] == '0'))

    # We can determine the init method automatically for Slurm.
    try:
        # Figure out the main host, and which rank we are.
        hostnames = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', node_list]
        )
        main_host = hostnames.split()[0].decode('utf-8')
        distributed_rank = int(os.environ['SLURM_PROCID'])
        device_id = int(os.environ['SLURM_LOCALID'])
        port = opt['port']
        print(
            'Initializing host {} as rank {}'
            .format(socket.gethostname(), distributed_rank)
        )
        # Begin distributed training
        multiprocess_train(distributed_rank, opt, port, device_id, main_host)
    except subprocess.CalledProcessError as e:
        # scontrol failed
        raise e
    except FileNotFoundError:
        # Slurm is not installed
        raise RuntimeError('SLURM does not appear to be installed.')


if __name__ == '__main__':
    main()
