# Copyright 2004-present Facebook. All Rights Reserved.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Simple loop which sets up a remote connection. The paired agent can run this
same loop but with the '--remote-host' flag set. For example...

Agent 1:
python remote.py

Agent 2:
python remote.py --remote-host --remote-address '*'

Now humans connected to each agent can communicate over that thread.
If you would like to use a model instead, merely set the '--model' flag:

Either Agent (or both):
python remote.py -m seq2seq
"""

from parlai.agents.remote_agent.remote_agent import RemoteAgentAgent
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import DialogPartnerWorld

import random

def main():
    random.seed(42)

    # Get command line arguments
    parser = ParlaiParser(True, True)
    RemoteAgentAgent.add_cmdline_args(parser)
    opt = parser.parse_args()

    if opt.get('model'):
        local = create_agent(opt)
    else:
        local = LocalHumanAgent(opt)
    remote = RemoteAgentAgent(opt)
    agents = [local, remote] if not opt['remote_host'] else [remote, local]
    world = DialogPartnerWorld(opt, agents)

    # Talk to the remote agent
    with world:
        while True:
            world.parley()

if __name__ == '__main__':
    main()
