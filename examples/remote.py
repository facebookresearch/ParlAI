#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simple loop which sets up a remote connection. The paired agent can run this
same loop but with the '--remote-host' flag set. For example...

Agent 1:
python remote.py

Agent 2:
python remote.py --remote-host

Now humans connected to each agent can communicate over that thread.


If you want to use this to feed a dataset to a remote agent, set the '--task':

Agent 1:
python remote.py -t "babi:task1k:1"


If you would like to use a model instead, merely set the '--model' flag:

Either Agent:
python remote.py -m seq2seq
"""

from parlai.agents.remote_agent.remote_agent import RemoteAgentAgent
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import DialogPartnerWorld, create_task

import random


def main():
    random.seed(42)

    # Get command line arguments
    parser = ParlaiParser(True, True)
    RemoteAgentAgent.add_cmdline_args(parser)
    opt = parser.parse_args()

    remote = RemoteAgentAgent(opt)
    if opt.get('task'):
        world = create_task(opt, [remote])
    else:
        if opt.get('model'):
            local = create_agent(opt)
        else:
            local = LocalHumanAgent(opt)
        # the remote-host goes **second**
        agents = [local, remote] if not opt['remote_host'] else [remote, local]
        world = DialogPartnerWorld(opt, agents)

    # Talk to the remote agent
    with world:
        while True:
            world.parley()
            print(world.display())


if __name__ == '__main__':
    main()
