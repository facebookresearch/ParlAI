#!/usr/bin/env python

import copy
import torch
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass
from torch.multiprocessing import Process
import torch.distributed as dist
import parlai.scripts.train_model as single_train


class DistributedProcess(Process):
    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['rank']
        super().__init__(daemon=True)

    def run(self):
        import builtins
        builtin_print = builtins.print
        msg = "[rank{:2d}]".format(self.opt['rank'])
        def new_print(*args, **kwargs):
            builtin_print(msg, *args, **kwargs)
        builtins.print = new_print
        print("Launching proc", self.rank)
        torch.cuda.set_device(self.opt['gpu'])
        print("set device")
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:21337",
            # or init_method=“file:///YOUR_HOME_DIR/file”
            world_size=self.opt['distributed_world_size'],
            rank=self.opt['rank'],
        )
        print("done ipg")
        single_train.TrainLoop(self.opt).train()


def main():
    parser = single_train.setup_args()
    parser.add_argument(
        '--distributed-world-size', type=int,
        default=torch.cuda.device_count(),
        help='Number of workers.'
    )
    opt = parser.parse_args()
    print("Staring with ", opt['distributed_world_size'])
    assert opt.get('numthreads') == 1

    opt_copies = []
    processes = []

    for rank in range(opt['distributed_world_size']):
        print("creating proc")
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
