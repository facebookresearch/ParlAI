#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Main launch script for single-host, multi-GPU training.

This is a drop-in replacement for train_model.py.  This script will launch N
subprocess, each which runs the full training loop independently.

Uses torch.nn.parallel.DistributedDataParallel for its main uses.  Agents must
specifically implement the wrapper of DistributedDatParallel, but all
TorchRankerAgents and TorchGeneratorAgents support this.
"""

import torch
import random
import copy
import os
import signal
import torch.distributed as dist
import parlai.scripts.train_model as single_train
import parlai.utils.distributed as distributed_utils


def multiprocess_train(
    rank, opt, port=61337, rank_offset=0, gpu=None, hostname='localhost'
):
    """
    Subprocess which initializes distributed training, and begins training.

    This should be launched n times for n GPUs; this is handled either in main
    or via srun.

    :param int rank: This process's rank - 1. (Starts at -1 ... n - 2). See comments.
    :param opt: command line options
    :param int port: A TCP port to use. This will need to be changed to run
        multiple distributed training setups on the same machine.
    :param int gpu: Which GPU to use. Defaults to using rank and local devices,
        but must be manually specified when using many-hosts.
    :param str hostname: Hostname of the main server.
    """
    # Set per-host options
    opt = copy.deepcopy(opt)
    # we need to manually adjust the rank differently in multiprocessing
    # and distributed train
    rank = rank + rank_offset
    opt['rank'] = rank
    if gpu is None:
        # default assumption is local GPUs
        gpu = rank % torch.cuda.device_count()
    opt['gpu'] = gpu
    # make sure we don't just use whatever GPU was saved in the model file
    if 'override' not in opt:
        opt['override'] = {}
    opt['override']['gpu'] = gpu

    # Suppress output of workers except the main host.
    if opt.get('verbose') or rank != 0:
        print_prefix = '[rank:{:3d}]'.format(rank)
    else:
        print_prefix = None
    suppress_output = not opt.get('verbose') and rank != 0

    with distributed_utils.override_print(suppress_output, print_prefix):
        # perform distributed setup, ensuring all hosts are ready
        torch.cuda.set_device(opt['gpu'])
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://{}:{}".format(hostname, port),
            world_size=opt['distributed_world_size'],
            rank=rank,
        )
        print("Distributed group initialized")

        # manual_seed can be a noop without this
        torch.cuda.init()
        # make sure all parameters will be in sync
        torch.manual_seed(42)
        # force a sync so that no one gets ahead, and all are seeded together
        distributed_utils.sync_object(None)

        # Run the actual training
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


def main():
    opt = setup_args().parse_args()
    port = random.randint(32000, 48000)
    return launch_and_train(opt, port)


if __name__ == '__main__':
    main()
