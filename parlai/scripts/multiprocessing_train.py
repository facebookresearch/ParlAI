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

import torch
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass
import os
import signal
import copy
from torch.multiprocessing import Process
import torch.distributed as dist
import parlai.scripts.train_model as single_train
import parlai.core.distributed_utils as distributed_utils


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        self.children_pids.append(pid)

    def error_listener(self):
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)


class DistributedProcess(Process):
    def __init__(self, opt, error_queue, port=41322):
        self.opt = opt
        self.rank = opt['rank']
        self.error_queue = error_queue
        self.port = port
        super().__init__(daemon=True)

    def run(self):
        try:
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
        except KeyboardInterrupt:
            # killed by parent, do nothing
            pass
        except Exception:
            import traceback
            self.error_queue.put((self.rank, traceback.format_exc()))


def main():
    parser = single_train.setup_args()
    parser.add_distributed_training_args()
    parser.set_defaults(distributed_world_size=torch.cuda.device_count())
    opt = parser.parse_args()

    mp = torch.multiprocessing.get_context('spawn')
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    opt_copies = []
    processes = []

    for rank in range(opt['distributed_world_size']):
        optc = copy.deepcopy(opt)
        opt_copies.append(optc)
        optc['rank'] = rank
        optc['gpu'] = rank % torch.cuda.device_count()
        p = DistributedProcess(optc, error_queue)
        processes.append(p)

    try:
        for p in processes:
            p.start()
            error_handler.add_child(p.pid)

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
