#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed training script. NOT MEANT TO BE CALLED DIRECTLY BY USER.

This script is meant to be in conjunction with
[SLURM ](https://slurm.schedmd.com/), which provides environmental
variables describing the environment.

An example sbatch script is below, for a 2-host, 8-GPU setup (16 total
gpus):

```bash\n\n
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
```
"""

import parlai.scripts.train_model as single_train
from parlai.core.script import ParlaiScript
import parlai.utils.distributed as distributed_utils


def setup_args():
    parser = single_train.setup_args()
    parser.add_distributed_training_args()
    parser.add_argument('--port', type=int, default=61337, help='TCP port number')
    return parser


class DistributedTrain(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        with distributed_utils.slurm_distributed_context(self.opt) as opt:
            self.train_loop = single_train.TrainLoop(opt)
            return self.train_loop.train()


if __name__ == '__main__':
    DistributedTrain.main()
