#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import signal
import unittest
import torch.distributed as dist
import torch
import pickle


def skipUnlessGPU(testfn, reason='Test requires a GPU'):
    gpu_available = torch.cuda.device_count() > 0
    return unittest.skipUnless(gpu_available, reason)(testfn)


def multiprocess_train(
    rank, opt, port=61337, rank_offset=0, gpu=None, hostname='localhost'
):

    rank = rank + rank_offset
    gpu = rank % torch.cuda.device_count()
    opt['gpu'] = gpu
    print(f'multiprocess_train: rank: {rank}, port: {port}, rank_offset: {rank_offset}')
    torch.cuda.set_device(opt['gpu'])
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://{}:{}".format(hostname, port),
        world_size=2,
        rank=rank,
    )
    print("Distributed group initialized")

    # manual_seed can be a noop without this
    torch.cuda.init()
    # make sure all parameters will be in sync
    torch.manual_seed(42)
    # force a sync so that no one gets ahead, and all are seeded together
    print('BEFORE SYNC')
    sync_object(None)
    print('AFTER SYNC')

    return None


def sync_object(data, max_size=16384):
    if not dist.is_available() or not dist.is_initialized():
        return data

    # prepare the buffer
    if not hasattr(sync_object, '_buffer') or sync_object._buffer.numel() < max_size:
        # cuda is safe because distributed mode is only okay with CUDA
        sync_object._buffer = torch.cuda.ByteTensor(max_size)

    buffer = sync_object._buffer

    if dist.get_rank() == 0:
        enc = pickle.dumps(data)
        enc_size = len(enc)
        if (enc_size + 2 > max_size) or (enc_size > 255 * 255):
            # can't store the size in the first 2 bytes
            raise ValueError('encoded data exceeds max_size')

        buffer[0] = enc_size // 255
        buffer[1] = enc_size % 255
        buffer[2 : enc_size + 2] = torch.ByteTensor(list(enc))

    dist.broadcast(buffer, 0)

    if dist.get_rank() > 0:
        # deserialize the data
        enc_size = buffer[0].item() * 255 + buffer[1].item()
        try:
            data = pickle.loads(bytes(buffer[2 : enc_size + 2].tolist()))
        except pickle.UnpicklingError:
            raise RuntimeError(
                'There was an unpickling error in sync_object. This likely '
                'means your workers got out of syncronization (e.g. one is '
                'expecting to sync and another is not.)'
            )

    return data


@skipUnlessGPU
class TestDistributed(unittest.TestCase):
    _base_config = dict(
        task='integration_tests:nocandidate',
        model='transformer/generator',
        optimizer='adamax',
        learningrate=7e-3,
        batchsize=32,
        validation_every_n_epochs=5,
        num_epochs=20,
        n_layers=1,
        n_heads=1,
        ffn_size=32,
        embedding_size=32,
        beam_size=1,
    )

    def _distributed_dummy_train(self):
        opt = {}
        port = 61337
        # Launch multiple subprocesses
        spawncontext = torch.multiprocessing.spawn(
            multiprocess_train,
            # need to give rank offset as 1 to cover the fact that the main
            # process is rank 0, but that spawn() doesn't let you control rank
            (opt, port, 1),
            nprocs=1,  # main proc will also run loop
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
        return None

    def test_distributed_dummy_train1(self):
        self._distributed_dummy_train()

    def test_distributed_dummy_train2(self):
        self._distributed_dummy_train()

    def setUp(self):
        print(f'[Setting up test {self._testMethodName}]')

    def tearDown(self):
        # we need to de-initialize the distributed world, otherwise other
        # tests will they're we're distributed when we're really not.
        dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
