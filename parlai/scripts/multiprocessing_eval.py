#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Main launch script for single-host, multi-GPU evaluation.

This is a drop-in replacement for eval_model.py.  This script will launch N
subprocess, each which runs the full eval loop independently.

Uses torch.nn.parallel.DistributedDataParallel for its main uses.  Agents must
specifically implement the wrapper of DistributedDataParallel, but all
TorchRankerAgents and TorchGeneratorAgents support this.
"""

import torch
import random
import os
import signal
import parlai.utils.distributed as distributed_utils
import parlai.scripts.eval_model as eval_model


def multiprocess_eval(
    rank, opt, port=61337, rank_offset=0, gpu=None, hostname='localhost'
):
    """
    Run a multiprocessing evaluation.

    Invoked by launch_and_eval, not instantiated directly.
    """
    with distributed_utils.distributed_context(
        rank, opt, port, rank_offset, gpu, hostname
    ) as opt:
        return eval_model.eval_model(opt)


def launch_and_eval(opt, port):
    """
    Perform a fork() to many processes.
    """
    # Launch multiple subprocesses
    spawncontext = torch.multiprocessing.spawn(
        multiprocess_eval,
        # need to give rank offset as 1 to cover the fact that the main
        # process is rank 0, but that spawn() doesn't let you control rank
        (opt, port, 1),
        nprocs=opt['distributed_world_size'] - 1,  # main proc will also run loop
        join=False,
    )

    try:
        retval = multiprocess_eval(0, opt, port)
        spawncontext.join()
        return retval
    except KeyboardInterrupt:
        # tell the subprocesses to stop too
        for p in spawncontext.processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGINT)
        raise


def setup_args():
    parser = eval_model.setup_args()
    parser.add_distributed_training_args()
    parser.set_defaults(distributed_world_size=torch.cuda.device_count())
    return parser


def main():
    opt = setup_args().parse_args()
    port = random.randint(32000, 48000)
    return launch_and_eval(opt, port)


if __name__ == '__main__':
    main()
