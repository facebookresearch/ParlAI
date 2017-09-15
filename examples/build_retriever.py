# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Generates a retriever file from the training data."""

import copy
import os

from parlai.agents.ir_baseline.ir_retrieve import StringMatchRetrieverAgent
from parlai.core.params import ParlaiParser, str2class
from parlai.core.worlds import create_task


def build_retriever(opt):
    if not opt.get('retriever_file'):
        StringMatchRetrieverAgent.print_info(None, 'Tried to build retriever but `--retriever-file` is not set. Set ' +
              'this param to save the retriever.')
        return
    StringMatchRetrieverAgent.print_info(None, 'setting up retriever.')
    if os.path.isfile(opt['retriever_file']):
        # retriever already built
        StringMatchRetrieverAgent.print_info(None, "retriever already built.")
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
    ordered_opt['datatype'] = 'train:ordered:stream'
    ordered_opt['batchsize'] = 1
    world_retriever = create_task(ordered_opt, retriever)
    retriever.print_info('start building retriever...')
    world_retriever.auto_execute(True, opt.get('retriever_maxexs'))
    retriever.print_info('retriever built.')


def main():
    # Get command line arguments
    argparser = ParlaiParser()
    StringMatchRetrieverAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()
    build_retriever(opt)

if __name__ == '__main__':
    main()
