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
parlai multiprocessing_train -m transformer/generator --batchsize 16 --task convai2 --model-file /tmp/mymodel
```
"""

import torch
import os
import signal
import traceback
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
        try:
            return single_train.TrainLoop(opt).train()
        except Exception:
            import parlai.utils.logging as logging

            logging.critical(traceback.format_exc())
            logging.critical(
                f"Got the above exception on worker {rank + rank_offset}. "
                "This may cause hangs requiring manual killing of processes."
            )
            raise


def launch_and_train(opt, port=None):
    """
    Perform a fork() to many processes.
    """
    if port is None:
        port = distributed_utils.find_free_port()
    # Launch multiple subprocesses
    spawncontext = torch.multiprocessing.start_processes(
        multiprocess_train,
        # need to give rank offset as 1 to cover the fact that the main
        # process is rank 0, but that spawn() doesn't let you control rank
        (opt, port, 1),
        nprocs=opt['distributed_world_size'] - 1,  # main proc will also run loop
        join=False,
        start_method='spawn',
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
        argparser = setup_args()
        argparser.add_argument('--port', type=int, default=None)
        return argparser

    def run(self):
        if self.opt['port'] is None:
            port = None
        else:
            port = self.opt['port']
        return launch_and_train(self.opt, port)


if __name__ == '__main__':
    MultiProcessTrain.main()
