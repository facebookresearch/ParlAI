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

import builtins
import copy
import torch
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass
from torch.multiprocessing import Process
import torch.distributed as dist
import parlai.scripts.train_model as single_train


def __override_print(suppress_output=False, prefix=None):
    builtin_print = builtins.print

    def new_print(*args, **kwargs):
        if suppress_output:
            return
        elif prefix:
            builtin_print(prefix, *args, **kwargs)
        else:
            builtin_print(*args, **kwargs)

    builtins.print = new_print


class DistributedProcess(Process):
    def __init__(self, opt, port=41322):
        self.opt = opt
        self.rank = opt['rank']
        self.port = port
        super().__init__(daemon=True)

    def run(self):
        __override_print(
            suppress_output=not self.opt.get('verbose_workers'),
            prefix="[rank{:2d}]".format(self.opt['rank']),
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
    grp = parser.add_argument_group('Distributed Training')
    grp.add_argument(
        '--distributed-world-size', type=int,
        default=torch.cuda.device_count(),
        help='Number of workers.'
    )
    # TODO: use --debug or --verbose instead?
    grp.add_argument(
        '--verbose-workers', type='bool', default=False,
        help='Additional workers stay silent.',
        hidden=True,
    )
    opt = parser.parse_args()
    assert opt.get('numthreads') == 1

    opt_copies = []
    processes = []

    if 'train:stream' in opt['datatype'] or 'ordered' in opt['datatype']:
        raise ValueError(
            "You should not combine ordered streaming with distributed training "
            "because all workers will have exactly the same minibatches, "
            "defeating the purpose."
        )

    for rank in range(opt['distributed_world_size']):
        optc = copy.deepcopy(opt)
        opt_copies.append(optc)
        optc['rank'] = rank
        optc['gpu'] = rank % torch.cuda.device_count()
        p = DistributedProcess(optc)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
