#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


"""
Main launch script for single-host, multi-GPU training. Uses
torch.nn.parallel.DistributedDataParallel for its main uses.

This script will launch N subprocess, each which runs the full
training loop independently.
"""

import copy
import torch
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass
from torch.multiprocessing import Process
import torch.distributed as dist
import parlai.scripts.train_model as single_train
import parlai.core.distributed_utils as distributed_utils


class DistributedProcess(Process):
    def __init__(self, opt, port=41322):
        self.opt = opt
        self.rank = opt['rank']
        self.port = port
        super().__init__(daemon=True)

    def run(self):
        distributed_utils.override_print(
            prefix='[rank{:2d}]'.format(self.rank)
        )
        print("Launching Process #{}".format(self.rank))
        torch.cuda.set_device(self.opt['gpu'])
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:{}".format(self.port),
            world_size=self.opt['distributed_world_size'],
            rank=self.opt['rank'],
        )
        print("Distributed group initialized")
        single_train.TrainLoop(self.opt).train()


def main():
    parser = single_train.setup_args()
    parser.add_distributed_training_args()
    parser.set_defaults(distributed_world_size=torch.cuda.device_count())
    opt = parser.parse_args()

    opt_copies = []
    processes = []

    for rank in range(opt['distributed_world_size']):
        optc = copy.deepcopy(opt)
        opt_copies.append(optc)
        optc['rank'] = rank
        optc['gpu'] = rank % torch.cuda.device_count()
        p = DistributedProcess(optc)
        processes.append(p)

    try:
        for p in processes:
            p.start()

        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()


if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
