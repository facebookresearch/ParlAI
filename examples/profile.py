# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Run the pytorch profiler and prints the results.

For example, to make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:
`python examples/profile.py -t babi:task1k:1 -m seq2seq -e 0.1 --dict-file /tmp/dict`'
"""

import pdb
import torch
from train_model import setup_args as train_args
from train_model import main as train

def setup_args():
    parser = train_args()
    profile = parser.add_argument_group('Profiler Arguments')
    profile.add_argument('--use-nvprof', type='bool', default=False,
                         help='If True, uses nvprof (might incur high overhead'
                              ' and assumes that the whole process is running '
                              'inside nvprof), otherwise uses a custom '
                              'CPU-only profiler (with negligible overhead). '
                              'Default: False.')
    profile.add_argument('--trace-path', type=str, default=None,
                         help='A path of the CUDA checkpoint. If specified, it'
                              ' will be left unmodified after profiling '
                              'finishes, so it can be opened and inspected in '
                              'nvvp. Otherwise it will be created in a '
                              'temporary directory and removed after reading '
                              'the results.')
    profile.add_argument('--debug', type='bool', default=False,
                         help='If true, enter debugger at end of run.')
    return parser

def main(parser):
    opt = parser.parse_args()
    with torch.autograd.profiler.profile(
        use_nvprof=opt['use_nvprof'], trace_path=opt['trace_path']
    ) as prof:
        train(parser)
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
        print('`cpu()` prints out cpu-sorted list, '
              '`cuda()` prints cuda-sorted list')

        pdb.set_trace()

if __name__ == '__main__':
    main(setup_args())
