# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import sys
import time


class Predictor(object):
    """Provides functionality for setting up a running version of a model and
    requesting predictions from that model on live data.

    Note that this maintains no World state (does not use a World), merely
    providing the observation directly to the model and getting a response.

    This is limiting when it comes to certain use cases, but is
    """

    def __init__(self, args=None, **kwargs):
        """Initializes the predictor, setting up opt automatically if necessary.

        Args is expected to be in the same format as sys.argv: e.g. a list in
        the form ['--model', 'seq2seq', '-hs', 128, '-lr', 0.5].

        kwargs is interpreted by appending '--' to it and replacing underscores
        with hyphens, so 'dict_file=/tmp/dict.tsv' would be interpreted as
        '--dict-file /tmp/dict.tsv'.
        """
        from parlai.core.params import ParlaiParser
        from parlai.core.agents import create_agent

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
        if 'episode_done' not in observation:
            observation['episode_done'] = True
        self.agent.observe(observation)
        reply = self.agent.act()
        return reply


class Timer(object):
    """Computes elapsed time."""
    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


def round_sigfigs(x, sigfigs=4):
    if x == 0:
        return 0
    return round(x, -math.floor(math.log10(abs(x)) - sigfigs + 1))
