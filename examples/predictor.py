# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Parallel of `interactive.py`, a basic example which allows local human
keyboard input to talk to a trained model. The command-line functionality is
essentially the same, for example:
`wget https://s3.amazonaws.com/fair-data/parlai/_models/drqa/squad.mdl`
`python examples/predictor.py -m drqa -mf squad.mdl`

Then enter something like:
"Bob is Blue.\nWhat is Bob?"
as the user input (or in general for the drqa model, enter
a context followed by '\n' followed by a question all as a single input.)

However, this example demonstrates how to use the predictor module for this
example, which can use a set of arguments other than the command line to provide
model predictions from within a running program.

As shown in this example, just create a predictor with arguments which specify
the model (e.g. p = Predictor(model='drqa')), and the Predictor module will set
up the model and handle communication with it.
"""

from parlai.core.predictor import Predictor
from parlai.agents.local_human.local_human import LocalHumanAgent

import random
import sys

def main(arg):
    random.seed(42)

    # Get command line arguments
    p = Predictor(sys.argv[1:])
    interactive = LocalHumanAgent({})

    # Show some example dialogs:
    reply = {}
    while True:
        if reply:
            interactive.observe(reply)
        query = interactive.act()
        reply = p.predict(query)

if __name__ == '__main__':
    main(sys.argv)
