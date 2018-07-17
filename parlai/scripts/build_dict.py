# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Generates a dictionary file from the training data."""

from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser, str2class
from parlai.core.worlds import create_task
from parlai.core.utils import TimeLogger
import copy
import os
import sys

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    dict_loop = parser.add_argument_group('Dictionary Loop Arguments')
    dict_loop.add_argument('--dict-maxexs', default=-1, type=int,
        help='max number of examples to build dict on')
    dict_loop.add_argument('--dict-include-valid', default=False, type='bool',
        help='Include validation set in dictionary building for task.')
    dict_loop.add_argument('--dict-include-test', default=False, type='bool',
        help='Include test set in dictionary building for task.')
    dict_loop.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    partial, _ = parser.parse_known_args(nohelp=True)
    if vars(partial).get('dict_class'):
        str2class(vars(partial).get('dict_class')).add_cmdline_args(parser)
    else:
        DictionaryAgent.add_cmdline_args(parser)
    return parser

def build_dict(opt, skip_if_built=False):
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: should be passed opt not Parser ]')
        opt = opt.parse_args()
    if not opt.get('dict_file'):
        print('Tried to build dictionary but `--dict-file` is not set. Set ' +
              'this param so the dictionary can be saved.')
        return

    if skip_if_built and os.path.isfile(opt['dict_file']):
        # Dictionary already built, skip all loading or setup
        print("[ dictionary already built .]")
        return None

    if opt.get('dict_class'):
        # Custom dictionary class
        dictionary = str2class(opt['dict_class'])(opt)
    else:
        # Default dictionary class
        dictionary = DictionaryAgent(opt)

    if os.path.isfile(opt['dict_file']):
        # Dictionary already built, return loaded dictionary agent
        print("[ dictionary already built .]")
        return dictionary

    ordered_opt = copy.deepcopy(opt)
    cnt = 0
    # we use train set to build dictionary

    ordered_opt['numthreads'] = 1
    ordered_opt['batchsize'] = 1
    ordered_opt['image_mode'] = 'none'
    if ordered_opt['task'] == 'pytorch_teacher':
        pytorch_teacher_task = ordered_opt.get('pytorch_teacher_task', '')
        if pytorch_teacher_task != '':
            ordered_opt['task'] = pytorch_teacher_task

    datatypes = ['train:ordered:stream']
    if opt.get('dict_include_valid'):
        datatypes.append('valid:stream')
    if opt.get('dict_include_test'):
        datatypes.append('test:stream')
    cnt = 0
    for dt in datatypes:
        ordered_opt['datatype'] = dt
        world_dict = create_task(ordered_opt, dictionary)
        # pass examples to dictionary
        print('[ running dictionary over data.. ]')
        log_every_n_secs = opt.get('log_every_n_secs', -1)
        if log_every_n_secs <= 0:
            log_every_n_secs = float('inf')
        log_time = TimeLogger()
        while not world_dict.epoch_done():
            cnt += 1
            if cnt > opt['dict_maxexs'] and opt['dict_maxexs'] > 0:
                print('Processed {} exs, moving on.'.format(opt['dict_maxexs']))
                # don't wait too long...
                break
            world_dict.parley()
            if log_time.time() > log_every_n_secs:
                sys.stdout.write('\r')
                text, _log = log_time.log(cnt, max(opt.get('dict_maxexs',0),
                                                   world_dict.num_examples()))
                sys.stdout.write(text)
                sys.stdout.flush()

    dictionary.save(opt['dict_file'], sort=True)
    print('[ dictionary built with {} tokens ]'.format(len(dictionary)))
    return dictionary


if __name__ == '__main__':
    build_dict(setup_args().parse_args())
