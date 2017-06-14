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


class Predictor():

    def __init__(self, opt=None, *args, **kwargs):
        """Initializes the predictor, setting up opt automatically if necessary.

        Args is expected to be in the same format as sys.argv,
        """
        if opt is None:
            if args and type(args) == list:

            parser = ParlaiParser(True, True)
            parser.set_defaults(kwargs)


    def predict(self, observation, top_n=1):
        """From a ParlAI-standard observation dict, returns a prediction from
        the model.
        """
        pass


def main(arg):
    # Get command line arguments
    p = Predictor(args)
    interactive = LocalHumanAgent({})

    # Show some example dialogs:
    reply = {}
    while True:
        if reply:
            interactive.observe(reply)
        query = interative.act()
        reply = p.predict(query)

if __name__ == '__main__':
    main(sys.argv)
