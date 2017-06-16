# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Provides functionality for setting up a running version of a model and
requesting predictions from that model on live data.
"""

from parlai.core.params import ParlaiParser

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.agents.local_human.local_human import LocalHumanAgent

import sys


class Predictor(object):

    def __init__(self, args=None, **kwargs):
        """Initializes the predictor, setting up opt automatically if necessary.
        Args is expected to be in the same format as sys.argv.
        """
        if args is None:
            args = []
        for k, v in kwargs.items():
            args.append('--' + str(k).replace('_', '-'))
            args.append(str(v))
        parser = ParlaiParser(True, True, model_argv=args)
        self.opt = parser.parse_args(args)
        self.agent = create_agent(self.opt)

    def predict(self, observation):
        """From a ParlAI-standard observation dict, returns a prediction from
        the model.
        """
        self.agent.observe(observation)
        reply = self.agent.act()
        return reply
