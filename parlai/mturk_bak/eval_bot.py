# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Contains basic functionality for setting up a simple MTurk bot evaluation.
In this example, a bot will be paired with a human, given the default
instructions and opening message, and then will chat with the bot.
"""

from parlai.core.agents import create_agent_from_shared

def setup_relay(num_hits, message):
    """Sets up relay server and returns a relay_server object which can poll
    HITS and send messages, as well as the range of HITS which should be polled.
    """
    # set up relay server
    return relay_server, range(num_hits)

def create_hits(opt, bot, num_hits, message=None):
    shared = bot.share()
    bots = [create_agent_from_shared(shared) for _ in range(num_hits)]

    relay_server, hit_ids = setup_relay(num_hits, message)
    hid_map = {hid: i for i, hid in enumerate(hid_ids)}

    hits_remaining = set(hit_ids)

    while len(hits_remaining) > 0:
        pops = []
        for hid in hits_remaining:
            reply = relay_server.poll(hid)
            if reply:
                agent = self.bots[hid_map[hid]]
                # observe could be in the else block?
                agent.observe(reply)
                if reply.get('done', False):
                    # we're done here
                    pops.append(hid)
                else:
                    # agent still needs to reply
                    relay_server.send(agent.act())
        for hid in pops:
            hits_remaining.remove(hid)
