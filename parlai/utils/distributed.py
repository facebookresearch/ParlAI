#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Useful utilities for training in distributed mode.

Many of these functions act as wrappers which perform no-ops if code is running in non-
distributed mode.
"""

import builtins
import copy
import os
import pickle
import contextlib
import subprocess
import socket
import parlai.utils.logging as logging

try:
    import torch.nn
    import torch.version
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def is_distributed():
    """
    Return if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()


def num_workers():
    """
    Get the total number of workers.
    """
    if not is_distributed():
        return 1
    else:
        return dist.get_world_size()


def is_primary_worker():
    """
    Determine if we are the primary (rank 0)  worker.

    Returns False if we are a secondary worker. Returns True if we are either (1) not in
    distributed mode (2) or are the primary (rank 0) worker.
    """
    return not is_distributed() or dist.get_rank() == 0


def get_rank():
    """
    Returns the rank of the current worker.

    Returns 0 if not in distributed.
    """
    if not is_distributed():
        return 0
    else:
        return dist.get_rank()


@contextlib.contextmanager
def override_print(suppress=False, prefix=None):
    """
    Context manager to override the print to suppress or modify output.

    Recommended usage is to call this with suppress=True for all non-primary
    workers, or call with a
    prefix of rank on all workers.

    >>> with override_print(prefix="rank{}".format(rank)):
    ...     my_computation()
    :param bool suppress:
        if true, all future print statements are noops.
    :param str prefix:
        if not None, this string is prefixed to all future print statements.
    """
    builtin_print = builtins.print

    def new_print(*args, **kwargs):
        if suppress:
            # do nothing
            return
        elif prefix:
            return builtin_print(prefix, *args, **kwargs)
        else:
            # default to normal print
            return builtin_print(*args, **kwargs)

    if prefix:
        logging.logger.add_format_prefix(prefix)
    if suppress:
        logging.disable()

    # override the print for now
    builtins.print = new_print
    yield
    # bring it back at the end of the context
    builtins.print = builtin_print

    if suppress:
        logging.enable()


def all_gather_list(data):
    """
    Gather arbitrary data from all nodes into a list.

    Similar to `~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    :param data:
        data from the local worker to be gathered on other workers

    :returns:
        a list containing [data1, data2, ...] of all workers
    """
    if not is_distributed():
        # fall back to just keeping things basic if we're not distributed
        return [data]

    # stolen shamelessly from fairseq
    # https://github.com/pytorch/fairseq/blob/c37250ab1c845919af721cd3f5c4cec2993aefe1/fairseq/distributed_utils.py#L116-L170
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    enc = list(pickle.dumps(data))
    enc_size = len(enc)

    # find the sizes of all the serialized items
    sizes = torch.zeros(world_size, dtype=torch.long).cuda()
    sizes[rank] = enc_size
    dist.all_reduce(sizes)

    # need to know our positions
    sizes = sizes.cpu()
    positions = sizes.cumsum(dim=0)

    buffer_size = positions[-1].item()
    buffer = torch.cuda.ByteTensor(buffer_size).zero_()

    start = positions[rank] - enc_size
    end = positions[rank]
    buffer[start:end] = torch.ByteTensor(enc)

    dist.all_reduce(buffer)

    result = []
    for i in range(world_size):
        out_buffer = buffer[positions[i] - sizes[i] : positions[i]]
        try:
            result.append(pickle.loads(bytes(out_buffer.tolist())))
        except pickle.UnpicklingError:
            raise RuntimeError(
                'There was an unpickling error in all_gather_list. This likely '
                'means your workers got out of synchronization (e.g. one is '
                'expecting to sync and another is not.)'
            )

    return result


def sync_object(data):
    """
    Sync an object among all workers.

    All workers will return the same value for `data` when returning from this
    method, always using the primary worker's version. Useful for ensuring control
    flow decisions are made the same.

    :param object data:
        The object to synchronize. Must be pickleable.

    :return: the synchronized data
    """
    value = all_gather_list(data if get_rank() == 0 else None)[0]
    return value


def sync_parameters(model: torch.nn.Module) -> bool:
    """
    Sync all parameters across all workers are the same.

    Always returns True, or raises an AssertionError if there was a failure.

    :param model: A pytorch model.
    :return: always True
    """
    if not is_distributed():
        # if things aren't distributed, of course things are in sync
        return True

    # sync all the parameters
    with torch.no_grad():
        for p in model.parameters():
            if not is_primary_worker():
                # zero out parameters on all workers EXCEPT the primary worker
                p.data.zero_()
            # sum the parameters across all workers, resulting in everyone having
            # the parameters of the primary worker
            dist.all_reduce(p.data, dist.ReduceOp.SUM)

    # double check everything synced correctly
    norm2 = sum((p.data**2).sum().float().item() for p in model.parameters())
    all_versions = all_gather_list(norm2)
    if not all(n == norm2 for n in all_versions):
        raise AssertionError(
            "Some models parameters were out of sync. Got the following norms: {}".format(
                " ".join(str(x) for x in all_versions)
            )
        )

    return True


@contextlib.contextmanager
def distributed_context(
    rank, opt, rank_offset=0, gpu=None, init_method="tcp://localhost:61337"
):
    """
    A context which wraps initialization of a distributed/multiprocessing run.

    Every process in the distributed run should launch with this. In true
    distributed setting you may wish to use slurm_distributed_context instead.

    :param int rank:
        This process's rank, less rank_offset.
    :param int rank_offset:
        Used as an offset of rank. Used between multiprocessing vs true distributed,
        and a hack around torch.multiprocessing.spawn being only used for the
        non-primary workers.
    :param opt:
        command line options
        distributed training setups on the same machine.
    :param int gpu:
        Which GPU to use. Defaults to using rank and local devices, but must be
        manually specified when using many-hosts.
    :param str init method:
        Init method, such as ``tcp://localhost:61337``. See torch.distributed docs.
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
        print_prefix = 'rank:{:3d} |'.format(rank)
    else:
        print_prefix = None
    suppress_output = not opt.get('verbose') and rank != 0

    with override_print(suppress_output, print_prefix):
        # perform distributed setup, ensuring all hosts are ready
        if opt['gpu'] != -1:
            torch.cuda.set_device(opt['gpu'])
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=opt['distributed_world_size'],
            rank=rank,
        )
        logging.info("Distributed group initialized")

        # manual_seed can be a noop without this
        torch.cuda.init()
        # make sure all parameters will be in sync
        torch.manual_seed(42)
        # force a sync so that no one gets ahead, and all are seeded together
        sync_object(None)

        try:
            yield opt
        finally:
            dist.destroy_process_group()


def get_dist_group():
    """
    Find the default pytorch distributed group.

    Used within FSDP to mark which workers are participating. Important to manually call
    this because FSDP will cache old groups, but our test suite will instantiate new
    groups per test.
    """
    from torch.distributed.distributed_c10d import _get_default_group

    return _get_default_group()


@contextlib.contextmanager
def slurm_distributed_context(opt):
    """
    Initialize a distributed context, using the SLURM environment.

    Does some work to read the environment to find a list of participating nodes
    and the main node.

    :param opt:
        Command line options.
    """
    # We can determine the init method automatically for Slurm.
    # double check we're using SLURM
    node_list = os.environ.get('SLURM_JOB_NODELIST')
    if node_list is None:
        raise RuntimeError(
            'Does not appear to be in a SLURM environment. '
            'You should not call this script directly; see launch_distributed.py'
        )
    try:
        # Figure out the main host, and which rank we are.
        hostnames = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', node_list]
        )
    except FileNotFoundError as e:
        # Slurm is not installed
        raise RuntimeError(
            f'SLURM does not appear to be installed. Missing file: {e.filename}'
        )

    main_host = hostnames.split()[0].decode('utf-8')
    distributed_rank = int(os.environ['SLURM_PROCID'])
    if opt.get('model_parallel'):
        # -1 signals to multiprocessing_train to use all GPUs available.
        # (A value of None signals to multiprocessing_train to use the GPU
        # corresponding to the rank.
        device_id = -1
    else:
        device_id = int(os.environ['SLURM_LOCALID'])
    port = opt['port']
    logging.info(
        f'Initializing host {socket.gethostname()} as rank {distributed_rank}, '
        f'main is {main_host}'
    )
    # Begin distributed training
    with distributed_context(
        distributed_rank, opt, 0, device_id, init_method=f"tcp://{main_host}:{port}"
    ) as opt:
        yield opt


def find_free_port() -> int:
    """
    Find a free port we can bind to locally.

    Credit: https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
