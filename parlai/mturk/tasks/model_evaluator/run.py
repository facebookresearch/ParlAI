# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.model_evaluator.worlds import ModelEvaluatorWorld
from parlai.mturk.core.agents import MTurkAgent
from task_config import task_config
import time
import os


def main():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()

    # The dialog model we want to evaluate
    from parlai.agents.ir_baseline.ir_baseline import IrBaselineAgent
    IrBaselineAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.getcwd())
    model_agent = IrBaselineAgent(opt=opt)

    # The task that we will evaluate the dialog model on
    task_opt = {}
    task_opt['datatype'] = 'test'
    task_opt['datapath'] = opt['datapath']
    task_opt['task'] = '#MovieDD-Reddit'

    # Create the MTurk agent which provides a chat interface to the Turker
    opt.update(task_config)
    mturk_agent_id = 'Worker'
    opt['agent_id'] = mturk_agent_id
    opt['mturk_agent_ids'] = [mturk_agent_id]
    opt['all_agent_ids'] = [ModelEvaluatorWorld.evaluator_agent_id, mturk_agent_id]
    opt['conversation_id'] = str(int(time.time()))

    mturk_agent = MTurkAgent(opt=opt)

    world = ModelEvaluatorWorld(opt=opt, model_agent=model_agent, task_opt=task_opt, mturk_agent=mturk_agent)

    while not world.episode_done():
        world.parley()

    world.shutdown()

if __name__ == '__main__':
    main()
