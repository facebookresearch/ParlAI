#!/usr/bin/env python

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
try:
    # We need to run this *very first*, but subprocesses will throw an
    # exception when running it
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass
import random
import copy
import os
import signal
import torch.distributed as dist
import parlai.scripts.train_model as single_train
import parlai.core.distributed_utils as distributed_utils


def multiprocess_train(rank, opt, port=61337, gpu=None, hostname='localhost'):
    """
    Subprocess which initializes distributed training, and begins training.

    This should be launched n times for n GPUs; this is handled either in main
    or via srun.

    :param int rank: This process's rank
    :param opt: command line options
    :param int port: A TCP port to use. This will need to be changed to run
        multiple distributed training setups on the same machine.
    :param int gpu: Which GPU to use. Defaults to using rank and local devices,
        but must be manually specified when using many-hosts.
    :param str hostname: Hostname of the main server.
    """
    # Set per-host options
    opt = copy.deepcopy(opt)
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
        print_prefix = '[rank:{:2d}]'.format(rank)
    else:
        print_prefix = None
    distributed_utils.override_print(
        suppress=(not opt.get('verbose') and rank != 0),
        prefix=print_prefix
    )

    # perform distributed setup, ensuring all hosts are ready
    torch.cuda.set_device(opt['gpu'])
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://{}:{}".format(hostname, port),
        world_size=opt['distributed_world_size'],
        rank=rank,
    )
    print("Distributed group initialized")

    # Run the actual training
    return single_train.TrainLoop(opt).train()


def main():
    parser = single_train.setup_args()
    parser.add_distributed_training_args()
    parser.set_defaults(distributed_world_size=torch.cuda.device_count())
    opt = parser.parse_args()

    port = random.randint(32000, 48000)

    # Launch multiple subprocesses
    spawncontext = torch.multiprocessing.spawn(
        multiprocess_train,
        (opt, port),
        nprocs=opt['distributed_world_size'],
        join=False,
    )

    try:
        spawncontext.join()
    except KeyboardInterrupt:
        # tell the subprocesses to stop too
        for p in spawncontext.processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGINT)


if __name__ == '__main__':
    main()
