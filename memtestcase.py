#!/usr/bin/env python

#SBATCH --gres=gpu:2
#SBATCH --job-name=distributed_example
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=10

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as td
import torch.nn.parallel as tp

START_BS = 8 * 1024

# these don't matter, just constants meant to be a "big" model
INPUT_SIZE = 8192
HID_SIZE = 4096
LAYERS = 8
OUT_CLASSES = 4


def wrap_dp(model):
    return tp.DataParallel(model)


def wrap_ddp_multi(model):
    td.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:61337',
        rank=0,
        world_size=1
    )
    model = tp.DistributedDataParallel(
        model,
        device_ids=None,
        broadcast_buffers=False,
    )
    return model


def wrap_ddp_single(model):
    td.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:61337',
        rank=0,
        world_size=1
    )
    model = tp.DistributedDataParallel(
        model,
        device_ids=[0],
        broadcast_buffers=False,
    )
    return model


def create_model(args):
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, HID_SIZE),
        nn.ReLU(),
    )
    for i in range(LAYERS):
        model.add_module('hidd' + str(i), nn.Linear(HID_SIZE, HID_SIZE))
        model.add_module('relu' + str(i), nn.ReLU())
    model.add_module('output', nn.Linear(HID_SIZE, OUT_CLASSES))
    return model


def fwbw(model, bs):
    print('  Forward with bs  = {:-6d}'.format(bs))
    X = torch.randn(bs, INPUT_SIZE).cuda()
    torch.cuda.synchronize()
    yhat = model(X)
    torch.cuda.synchronize()
    loss = yhat.sum()
    torch.cuda.synchronize()
    print('  Backward with bs = {:-6d}'.format(bs))
    loss.backward()
    torch.cuda.synchronize()
    model.zero_grad()
    torch.cuda.synchronize()


def run_trial(args):
    print('Conda PREFIX:', os.environ['CONDA_PREFIX'])
    print('Torch version:', torch.version.__version__)
    print('CUDA version:', torch.version.cuda)

    model = create_model(args).cuda()
    if args.mode == 'dp':
        print('Wrapping in DataParallel')
        model = wrap_dp(model)
    elif args.mode == 'ddp_multi':
        print('Wrapping in DistributedDataParallel (equiv to 1 proc per node)')
        model = wrap_ddp_multi(model)
    elif args.mode == 'ddp_single':
        print('Using a single GPU in distributed (equiv to 1 proc per gpu)')
        torch.cuda.set_device(0)
        model = wrap_ddp_single(model)
    elif args.mode == 'single':
        print('Using a single GPU')
        pass
    else:
        raise ValueError('--mode wrong')

    bs = args.bs
    times_oomed = 0
    while times_oomed < args.ooms:
        # continuously double the batch size until we OOM
        try:
            print('Step bs=', bs)
            fwbw(model, bs)
            print('FW/BW succeeded. Doubling BS')
            bs *= 2
        except RuntimeError as rerr:
            if 'memory' not in str(rerr):
                # not the exception we wanted
                raise rerr
            # okay, we found the memory error! Now try to run a NOOP pass
            # for DDP nodes. Production example here:
            # https://github.com/pytorch/fairseq/blob/3658fa329b8cb987d951b2e38ec86c44b9e1fea5/fairseq/trainer.py#L361-L368
            times_oomed += 1
            print('OOM #{}! Running through a tiny batch to catch up worker'.format(times_oomed))
            fwbw(model, 2)
            print('Succeeded on the oom batch.')
            # start the doubling procedure again
            bs = args.bs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', default='ddp', choices=('dp', 'ddp_multi', 'ddp_single', 'single'),
        help='DataParallel, DistributedDataParallel, or single gpu'
    )
    parser.add_argument(
        '--ooms', default=1, type=int,
        help='Number of times to OOM'
    )
    parser.add_argument(
        '--bs', default=START_BS, type=int,
        help='Initial batch size',
    )
    args = parser.parse_args()
    run_trial(args)
    print('Test passed.')


if __name__ == '__main__':
    main()
