#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.worlds import World


# ----- Baseline overworld that simply defers to the default world ----- #
class SimpleMessengerOverworld(World):
    """Passthrough world to spawn task worlds of only one type

    Demos of more advanced overworld functionality exist in the overworld demo
    """

    def __init__(self, opt, agent):
        self.agent = agent
        self.opt = opt

    def return_overworld(self):
        pass

    def parley(self):
        return 'default'


class OnboardWorld(World):
    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False

    @staticmethod
    def run(opt, agent, task_id):
        pass

    def parley(self):
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        pass
