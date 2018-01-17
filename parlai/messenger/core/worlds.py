# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.worlds import World


class MessengerOnboardWorld(World):
    """Generic world for onboarding a new person and collecting
    information from them."""
    def __init__(self, opt, messenger_agent):
        self.messenger_agent = messenger_agent
        self.episodeDone = False

    def parley(self):
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        pass


class MessengerTaskWorld(World):
    """Generic world for Messenger tasks."""
    def __init__(self, opt, messenger_agent):
        self.messenger_agent = messenger_agent
        self.episodeDone = False

    def parley(self):
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
        self.messenger_agent.shutdown()
        """
        Use the following code if there are multiple messenger agents:

        global shutdown_agent
        def shutdown_agent(messenger_agent):
            messenger_agent.shutdown()
        Parallel(
            n_jobs=len(self.messenger_agents),
            backend='threading'
        )(delayed(shutdown_agent)(agent) for agent in self.messenger_agents)
        """
