#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
from parlai.core.torch_agent import Batch
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import _create_task_agents
from torch.utils.data import IterableDataset, DataLoader


class ParlaiIterableDataset(IterableDataset):
    def __init__(self, opt, agent):
        super().__init__()
        self.opt = opt
        self.agent = agent
        self.agent.use_cuda = False
        if 'train' in opt['datatype'] and 'ordered' not in opt['datatype']:
            opt['datatype'] = opt['datatype'] + ':ordered'
        task_agent = _create_task_agents(opt)[0]
        self.episodes = []
        acts = []
        while not task_agent.epoch_done():
            act = task_agent.act()
            acts.append(act)
            if act['episode_done']:
                self.episodes.append(acts)
                acts = []
        if acts:
            self.episodes.append(acts)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            indices = range(len(self.episodes))
        else:
            indices = range(worker_info.id, len(self.episodes), worker_info.num_workers)

        indices = list(indices)
        random.shuffle(indices)
        episode_iters = [iter(self.episodes[i]) for i in indices]

        batchsize = self.opt['batchsize']
        batch = []
        while episode_iters and len(batch) < batchsize:
            batch.append(episode_iters.pop(0))
        agents = [self.agent.clone() for _ in batch]

        while batch:
            output = []
            for i in reversed(range(len(batch))):
                try:
                    output.append(agents[i].observe(next(batch[i])))
                    agents[i].self_observe({})
                except StopIteration:
                    if episode_iters:
                        batch[i] = episode_iters.pop(0)
                        output.append(agents[i].observe(next(batch[i])))
                        agents[i].self_observe({})
                    else:
                        batch.pop(i)
                        agents.pop(i)

            yield self.agent.batchify(output)


def main():
    pp = ParlaiParser(True, True)
    opt = pp.parse_args()
    agent = create_agent(opt)
    dl = DataLoader(
        ParlaiIterableDataset(opt, agent.clone()),
        batch_size=None,
        num_workers=4,
        pin_memory=False,
    )
    total = 0
    for batch in dl:
        batch = Batch(**batch).cuda()
        total += batch['batchsize']
        print("batch: ", total, type(batch))
        # batch.cuda()


if __name__ == '__main__':
    main()
