#!/usr/bin/env python3

import sys
import pdb
import queue
import torch
import time
import random
import torch.multiprocessing as multiprocessing

from parlai.core.torch_agent import Batch
from parlai.core.worlds import _create_task_agents
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser

NUM_EXAMPLES = 200
NUM_WORKERS = 10
BS = 32
SEQ = 128


class ForkedPdb(pdb.Pdb):
    """
    A Pdb subclass that may be used from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class End:
    pass


def prep(agent, acts):
    batch = agent.batchify(acts)
    # del batch['observations']
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].share_memory_()
    return dict(batch)


def generate_tensors(worker_id, queue, evt, opt, agent):
    print(f"Starting generate")
    agent.use_cuda = False
    if 'train' in opt['datatype'] and 'ordered' not in opt['datatype']:
        opt['datatype'] = opt['datatype'] + ':ordered'
    task_agent = _create_task_agents(opt)[0].clone()

    act_count = 0
    acts = []
    while not task_agent.epoch_done():
        act = task_agent.act()
        if act['episode_done']:
            act_count += 1
        if act_count % NUM_WORKERS != worker_id:
            continue
        acts.append(agent.observe(act))
        agent.self_observe({})
        if len(acts) == BS:
            queue.put((worker_id, prep(agent, acts)))
            acts = []
    if acts:
        queue.put((worker_id, prep(agent, acts)))

    queue.put((worker_id, End()))
    evt.wait()


class Reader(object):
    def __init__(self, opt):
        mp = multiprocessing.get_context("fork")
        self.queue = mp.Queue(1000)
        self.events = [mp.Event() for _ in range(NUM_WORKERS)]
        self.agent = create_agent(opt)
        self.processes = [
            mp.Process(
                target=generate_tensors,
                args=(i, self.queue, self.events[i], opt, self.agent),
            )
            for i in range(NUM_WORKERS)
        ]
        self.i = 0
        self.running_workers = 0
        for p in self.processes:
            p.start()
            self.running_workers += 1
        print("Done initializing")

    def _finish(self, worker_id):
        self.running_workers -= 1

    def __del__(self):
        for e in self.events:
            e.set()
        for p in self.processes:
            p.terminate()

    def parley(self):
        example = None
        while example is None:
            worker_id, example = self.queue.get()
            if isinstance(example, End):
                self._finish(worker_id)
                if self.running_workers == 0:
                    raise StopIteration
                example = None
        batch = Batch(**example)
        batch.cuda()
        self.i += batch.batchsize
        if self.i % 10 == 0:
            print(self.i)


def main():
    opt = ParlaiParser(True, True).parse_args()
    r = Reader(opt)

    while True:
        try:
            r.parley()
        except StopIteration:
            break
        except KeyboardInterrupt:
            del r
    print(r.i)


if __name__ == '__main__':
    main()
