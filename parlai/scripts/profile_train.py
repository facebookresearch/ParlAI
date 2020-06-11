#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Run the python or pytorch profiler and prints the results.

Examples
--------

To make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:

.. code-block:: shell

  python examples/profile.py -t babi:task1k:1 -m seq2seq -e 0.1 --dict-file /tmp/dict
"""

from parlai.core.params import ParlaiParser
from parlai.scripts.train_model import setup_args as train_args
from parlai.scripts.train_model import TrainLoop
import parlai.utils.logging as logging
import cProfile
import io
import pdb
import pstats

try:
    import torch
except ImportError:
    logging.error('Torch not found -- only cProfile allowed with this tool.')


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'cProfile a training run')
    parser = train_args(parser)
    profile = parser.add_argument_group('Profiler Arguments')
    profile.add_argument(
        '--torch',
        type='bool',
        default=False,
        help='If true, use the torch profiler. Otherwise use cProfile.',
    )
    profile.add_argument(
        '--torch-cuda',
        type='bool',
        default=False,
        help='If true, use the torch cuda profiler. Otherwise use cProfile.',
    )
    profile.add_argument(
        '--debug',
        type='bool',
        default=False,
        help='If true, enter debugger at end of run.',
    )
    return parser


def profile(opt):
    if isinstance(opt, ParlaiParser):
        logging.error('profile should be passed opt not Parser')
        opt = opt.parse_args()
    if opt['torch'] or opt['torch_cuda']:
        with torch.autograd.profiler.profile(use_cuda=opt['torch_cuda']) as prof:
            TrainLoop(opt).train()
        print(prof.total_average())

        sort_cpu = sorted(prof.key_averages(), key=lambda k: k.cpu_time)
        sort_cuda = sorted(prof.key_averages(), key=lambda k: k.cuda_time)

        def cpu():
            for e in sort_cpu:
                print(e)

        def cuda():
            for e in sort_cuda:
                print(e)

        cpu()

        if opt['debug']:
            print(
                '`cpu()` prints out cpu-sorted list, '
                '`cuda()` prints cuda-sorted list'
            )

            pdb.set_trace()
    else:
        pr = cProfile.Profile()
        pr.enable()
        TrainLoop(opt).train()
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        if opt['debug']:
            pdb.set_trace()


if __name__ == '__main__':
    profile(setup_args().parse_args())
