# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Generates a retriever file from the training data."""

import copy
import importlib
import os
from tqdm import tqdm

from parlai.agents.ir_baseline.ir_retrieve import StringMatchRetrieverAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser, str2class
from parlai.core.worlds import (
    DialogPartnerWorld,
    create_task,
    execute_world,
)

def build_retriever(opt):
    if not opt.get('retriever_file'):
        StringMatchRetrieverAgent.print_info('Tried to build retriever but `--retriever-file` is not set. Set ' +
              'this param to save the retriever.')
        return
    StringMatchRetrieverAgent.print_info('setting up retriever.')
    if os.path.isfile(opt['retriever_file']):
        # retriever already built
        StringMatchRetrieverAgent.print_info("retriever already built.")
        return
    if opt.get('retriever_class'):
        # Custom retriever class
        retriever = str2class(opt['retriever_class'])(opt)
    else:
        # Default retriever class
        retriever = StringMatchRetrieverAgent(opt)
    ordered_opt = copy.deepcopy(opt)
    cnt = 0
    # we use train set to build retriever
    ordered_opt['datatype'] = 'train:ordered'
    if 'stream' in opt['datatype']:
        ordered_opt['datatype'] += ':stream'
    # ordered_opt['numthreads'] = 1
    ordered_opt['batchsize'] = 1
    world_dict = create_task(ordered_opt, retriever)
    StringMatchRetrieverAgent.print_info('start building retriever...')
    execute_world(world_dict, use_tqdm=False, maxes=opt.get('retriever_maxexs'))
    StringMatchRetrieverAgent.print_info('retriever built.')
    retriever.save()


def main():
    # Get command line arguments
    argparser = ParlaiParser()
    DictionaryAgent.add_cmdline_args(argparser)
    StringMatchRetrieverAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()
    build_retriever(opt)

if __name__ == '__main__':
    main()
