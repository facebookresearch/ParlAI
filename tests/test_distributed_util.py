import torch
import torch.distributed as dist
from parlai.core.distributed_utils import DistributionConcatenation2D
import builtins
import random
import unittest
import time

WORLD_SIZE = 2

def launch(my_function):
    """ Launch a single function on 2 GPUS.
        this function should take 2 parameters: rank and world_size
    """
    port = random.randint(32000, 48000)
    spawncontext = torch.multiprocessing.spawn(
        init_and_launch,
        (WORLD_SIZE, my_function, port),
        nprocs=WORLD_SIZE,
        join=False,
    )

    try:
        spawncontext.join()
    except KeyboardInterrupt:
        # tell the subprocesses to stop too
        for p in spawncontext.processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGINT)

def override_print(suppress=False, prefix=None):
    """ Copied from ParlAI.
    """
    builtin_print = builtins.print
    def new_print(*args, **kwargs):
        if suppress:
            return
        elif prefix:
            return builtin_print(prefix, *args, **kwargs)
        else:
            return builtin_print(*args, **kwargs)
    builtins.print = new_print

def init_and_launch(rank, world_size, my_function, port=61337, gpu=None, hostname='localhost'):
    """ Copied from PARLAI
    """

    # Suppress output of workers except the main host.
    print_prefix = ""
    override_print(
        suppress=rank != 0,
        prefix=print_prefix
    )

    # perform distributed setup, ensuring all hosts are ready
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://{}:{}".format(hostname, port),
        world_size=WORLD_SIZE,
        rank=rank,
    )
    print("Distributed group initialized")
    my_function(rank, world_size)

class DistributionConcatenation2DTestModule(torch.nn.Module):
    """ An identity matrix followed by  DistributionConcatenation2D
    """
    def __init__(self, keep_self_first, dim):
        super(DistributionConcatenation2DTestModule, self).__init__()
        self.identity = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.identity.weight)
        self.dist_concat = DistributionConcatenation2D(keep_self_first)

    def forward(self, x):
        return self.dist_concat(self.identity(x))


def test_function_dist_concat(rank, world_size):
    # first test
    dim = 10
    sub_batch_size = 40
    torch.cuda.set_device(rank)
    test_module = DistributionConcatenation2DTestModule(False, dim).cuda()
    if dist.is_available() and dist.is_initialized():
        test_module = torch.nn.parallel.DistributedDataParallel(
            test_module,
            device_ids=[rank],
            broadcast_buffers=False,
        )

    value = rank + 1
    x = torch.zeros(sub_batch_size, dim).cuda() + value
    x.requires_grad=True
    y = test_module(x)
    assert y.size(0) == 2 * sub_batch_size, "Size of y is " % y.size(0)
    assert y[0,0] == 1
    assert y[-1,0] == 2
    loss = torch.sum(y) * value
    assert loss.item() == dim * sub_batch_size * 3 * value
    loss.backward()
    # the gradients of x should be 3 everywhere
    assert x.grad[0][0].item() == 3, \
        "The grad isn't right? %.3f" % x.grad[0][0].item()
    # however the gradients of the identity matrix should be 180 (120+240/2),
    # because DistributedDataParallel should have averaged them during
    # the backward()
    grad_identity = test_module.module.identity.weight.grad
    assert grad_identity[0,0].item() == 180, \
        "The grad isn't right? %.3f" % grad_identity[0][0].item()

    # second test, this time with keep self_first = true
    test_module = DistributionConcatenation2DTestModule(True, dim).cuda()
    if dist.is_available() and dist.is_initialized():
        test_module = torch.nn.parallel.DistributedDataParallel(
            test_module,
            device_ids=[rank],
            broadcast_buffers=False,
        )
    x = torch.zeros(sub_batch_size, dim).cuda() + value
    x.requires_grad=True
    y = test_module(x)
    assert y.size(0) == 2 * sub_batch_size, "Size of y is " % y.size(0)
    assert y[0,0] == value
    assert y[-1,0] == 3 - value
    return True

class TestDistConcat(unittest.TestCase):
    """Basic tests on the built-in parlai Dictionary."""

    def test_dist_concat_1(self):
        if torch.cuda.device_count() < 2:
            print("Sorry to execute this test you need to be on a machine with "
                  " at least 2 GPUs")
            return
        launch(test_function_dist_concat)

if __name__ == '__main__' :
    unittest.main()
