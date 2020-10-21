#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Main launch script for single-host, multi-GPU training.

This is a drop-in replacement for [train_model]. This script will
launch N subprocess, each which runs the full training loop
independently.

Uses torch.nn.parallel.DistributedDataParallel for its main uses. Agents
must specifically implement the wrapper of DistributedDatParallel, but
all TorchRankerAgents and TorchGeneratorAgents support this.

## Examples

```shell
parlai multiprocessing_train -m transformer/generator -bs 16 -t convai2 -mf /tmp/mymodel
```
"""

import torch
import random
import os
import signal
import parlai.scripts.train_model as single_train
import parlai.utils.distributed as distributed_utils
from parlai.core.script import ParlaiScript, register_script


def multiprocess_train(
    rank, opt, port=61337, rank_offset=0, gpu=None, hostname='localhost'
):
    init_method = f"tcp://{hostname}:{port}"
    with distributed_utils.distributed_context(
        rank, opt, rank_offset, gpu, init_method=init_method
    ) as opt:
        # Run the actual training
        opt['multiprocessing'] = True
        return single_train.TrainLoop(opt).train()


def launch_and_train(opt, port):
    """
    Perform a fork() to many processes.
    """
    # Launch multiple subprocesses
    spawncontext = torch.multiprocessing.spawn(
        multiprocess_train,
        # need to give rank offset as 1 to cover the fact that the main
        # process is rank 0, but that spawn() doesn't let you control rank
        (opt, port, 1),
        nprocs=opt['distributed_world_size'] - 1,  # main proc will also run loop
        join=False,
    )

    try:
        retval = multiprocess_train(0, opt, port)
        spawncontext.join()
        return retval
    except KeyboardInterrupt:
        # tell the subprocesses to stop too
        for p in spawncontext.processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGINT)
        raise


def setup_args():
    parser = single_train.setup_args()
    parser.add_distributed_training_args()
    parser.set_defaults(distributed_world_size=torch.cuda.device_count())
    return parser


@register_script("multiprocessing_train", aliases=["mp_train"], hidden=True)
class MultiProcessTrain(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        port = random.randint(32000, 48000)
        return launch_and_train(self.opt, port)


if __name__ == '__main__':
    MultiProcessTrain.main()
