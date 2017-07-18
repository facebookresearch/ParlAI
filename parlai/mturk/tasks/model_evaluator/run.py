# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.model_evaluator.worlds import ModelEvaluatorWorld
from parlai.mturk.core.agents import MTurkAgent, MTurkManager
from task_config import task_config
import time
import os
import copy
from itertools import product
from joblib import Parallel, delayed


def main():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()

    # The dialog model we want to evaluate
    from parlai.agents.ir_baseline.ir_baseline import IrBaselineAgent
    IrBaselineAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt.update(task_config)

    # The task that we will evaluate the dialog model on
    task_opt = {}
    task_opt['datatype'] = 'test'
    task_opt['datapath'] = opt['datapath']
    task_opt['task'] = '#MovieDD-Reddit'

    mturk_agent_id = 'Worker'
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids = [mturk_agent_id],
        all_agent_ids = [ModelEvaluatorWorld.evaluator_agent_id, mturk_agent_id] # In speaking order
    )
    mturk_manager.init_aws(opt=opt)
    
    global run_hit
    def run_hit(hit_index, assignment_index, opt, task_opt, mturk_manager):
        conversation_id = str(hit_index) + '_' + str(assignment_index)

        model_agent = IrBaselineAgent(opt=opt)
        # Create the MTurk agent which provides a chat interface to the Turker
        mturk_agent = MTurkAgent(id=mturk_agent_id, manager=mturk_manager, conversation_id=conversation_id, opt=opt)
        world = ModelEvaluatorWorld(opt=opt, model_agent=model_agent, task_opt=task_opt, mturk_agent=mturk_agent)

        while not world.episode_done():
            world.parley()
        world.shutdown()
        world.review_work()

    mturk_manager.create_hits(opt=opt)
    results = Parallel(n_jobs=opt['num_hits'] * opt['num_assignments'], backend='threading') \
                (delayed(run_hit)(hit_index, assignment_index, opt, task_opt, mturk_manager) \
                    for hit_index, assignment_index in product(range(1, opt['num_hits']+1), range(1, opt['num_assignments']+1)))    
    mturk_manager.shutdown()

if __name__ == '__main__':
    main()
