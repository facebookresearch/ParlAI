# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.utils import warn_once

warn_once(
    "Public release transformer models are currently in beta. The name of "
    "command line options may disappear before a stable release. We welcome "
    "your feedback. Please file feedback as issues at "
    "https://github.com/facebookresearch/ParlAI/issues/new"
)


def add_common_cmdline_args(argparser):
    argparser.add_argument('-esz', '--embedding-size', type=int, default=300,
                           help='Size of all embedding layers')
    argparser.add_argument('--n-layers', type=int, default=2)
    argparser.add_argument('--ffn-size', type=int, default=300,
                           help='Hidden size of the FFN layers')
    argparser.add_argument('--attention-dropout', type=float, default=0.0)
    argparser.add_argument('--relu-dropout', type=float, default=0.0)
    argparser.add_argument('--n-heads', type=int, default=3)
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)


class Transformer(Agent):
    def __init__(self, opt, shared):
        raise RuntimeError(
            "`--model transformer` is not a valid choice. Please select either "
            "`--model transformer/generator` or `--model transformer/generator"
        )
