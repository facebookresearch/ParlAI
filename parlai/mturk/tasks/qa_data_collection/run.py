# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.qa_data_collection.worlds import QADataCollectionWorld
from parlai.mturk.core.agents import MTurkAgent
from task_config import task_config
import time
import os
import importlib
import copy
try:
    from joblib import Parallel, delayed
except ModuleNotFoundError:
    raise SystemExit("Please install joblib by running: pip install joblib")

def main():
    global run_hit
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.getcwd())

    # Initialize a SQuAD teacher agent, which we will later get context from
    module_name = 'parlai.tasks.squad.agents'
    class_name = 'DefaultTeacher'
    my_module = importlib.import_module(module_name)
    task_class = getattr(my_module, class_name)
    task_opt = {}
    task_opt['datatype'] = 'train'
    task_opt['datapath'] = opt['datapath']

    # Create the MTurk agent which provides a chat interface to the Turker
    opt.update(task_config)
    mturk_agent_id = 'Worker'
    opt['agent_id'] = mturk_agent_id
    opt['mturk_agent_ids'] = [mturk_agent_id]
    opt['all_agent_ids'] = [QADataCollectionWorld.collector_agent_id, mturk_agent_id]
    opt['run_id'] = str(int(time.time()))

    def run_hit(i, task_class, task_opt, opt):
        task = task_class(task_opt)
        opt['conversation_id'] = str(i)
        mturk_agent = MTurkAgent(opt=opt)
        world = QADataCollectionWorld(opt=opt, task=task, mturk_agent=mturk_agent)
        while not world.episode_done():
            world.parley()
        world.shutdown()

    MTurkAgent.init_aws(opt)
    results = Parallel(n_jobs=opt['num_hits'], backend='threading')(delayed(run_hit)(i, task_class, task_opt, copy.deepcopy(opt)) for i in range(1, opt['num_hits']+1))
    MTurkAgent.review_hits(opt=opt)

if __name__ == '__main__':
    main()
