#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


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


def multiprocess_train(rank, opt, port):
    try:
        opt = copy.deepcopy(opt)
        opt['rank'] = rank
        opt['gpu'] = rank % torch.cuda.device_count()

        if opt.get('verbose') or rank != 0:
            print_prefix = '[rank:{:2d}]'.format(rank)
        else:
            print_prefix = None

        distributed_utils.override_print(
            suppress=(not opt.get('verbose') and rank != 0),
            prefix=print_prefix
        )
        print("Launching Process #{}".format(rank))
        torch.cuda.set_device(opt['gpu'])
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:{}".format(port),
            world_size=opt['distributed_world_size'],
            rank=rank,
        )
        print("Distributed group initialized")
        single_train.TrainLoop(opt).train()
    except KeyboardInterrupt:
        pass


def main():
    parser = single_train.setup_args()
    parser.add_distributed_training_args()
    parser.set_defaults(distributed_world_size=torch.cuda.device_count())
    opt = parser.parse_args()

    port = random.randint(32000, 48000)

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
