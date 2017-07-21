# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import os
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.agents import MTurkAgent, MTurkManager
from parlai.mturk.tasks.multi_agent_dialog.worlds import MTurkMultiAgentDialogWorld
from parlai.agents.local_human.local_human import LocalHumanAgent
from task_config import task_config
import copy
from itertools import product
from joblib import Parallel, delayed

"""
This task consists of two local human agents and two MTurk agents,
chatting with each other in a free-form format.
You can end the conversation by sending a message ending with
`[DONE]` from human_1.
"""
def main():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt.update(task_config)

    mturk_agent_1_id = 'mturk_agent_1'
    mturk_agent_2_id = 'mturk_agent_2'
    human_agent_1_id = 'human_1'
    human_agent_2_id = 'human_2'
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids = [mturk_agent_1_id, mturk_agent_2_id]
    )
    mturk_manager.init_aws(opt=opt)
    mturk_manager.start_new_run(opt=opt)

    global run_hit
    def run_hit(hit_index, assignment_index, opt, mturk_manager):
        # Create mturk agents
        mturk_agent_1 = MTurkAgent(id=mturk_agent_1_id, manager=mturk_manager, hit_index=hit_index, assignment_index=assignment_index, opt=opt)
        mturk_agent_2 = MTurkAgent(id=mturk_agent_2_id, manager=mturk_manager, hit_index=hit_index, assignment_index=assignment_index, opt=opt)

        # Create the local human agents
        human_agent_1 = LocalHumanAgent(opt=None)
        human_agent_1.id = human_agent_1_id
        human_agent_2 = LocalHumanAgent(opt=None)
        human_agent_2.id = human_agent_2_id

        world = MTurkMultiAgentDialogWorld(opt=opt, agents=[human_agent_1, human_agent_2, mturk_agent_1, mturk_agent_2])

        while not world.episode_done():
            world.parley()
        world.shutdown()

    mturk_manager.create_hits(opt=opt)
    results = Parallel(n_jobs=opt['num_hits'] * opt['num_assignments'], backend='threading') \
                (delayed(run_hit)(hit_index, assignment_index, opt, mturk_manager) \
                    for hit_index, assignment_index in product(range(1, opt['num_hits']+1), range(1, opt['num_assignments']+1)))
    mturk_manager.shutdown()

if __name__ == '__main__':
    main()
