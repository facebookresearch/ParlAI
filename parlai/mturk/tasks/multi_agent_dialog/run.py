# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import os
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.agents import MTurkAgent, MTurkManager
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.core.worlds import MultiAgentDialogWorld
from task_config import task_config
import copy
try:
    from joblib import Parallel, delayed
except ModuleNotFoundError:
    raise SystemExit("Please install joblib by running: pip install joblib")

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
    opt['task'] = os.path.basename(os.getcwd())
    opt.update(task_config)

    global run_hit
    def run_hit(i, opt, mturk_manager):
        # Create mturk agents
        mturk_agent_1 = MTurkAgent(id='mturk_agent_1', manager=mturk_manager, conversation_id=i, opt=opt)
        mturk_agent_2 = MTurkAgent(id='mturk_agent_2', manager=mturk_manager, conversation_id=i, opt=opt)

        # Create the local human agents
        human_agent_1 = LocalHumanAgent(opt=None)
        human_agent_1.id = 'human_1'
        human_agent_2 = LocalHumanAgent(opt=None)
        human_agent_2.id = 'human_2'

        world = MultiAgentDialogWorld(opt=opt, agents=[human_agent_1, human_agent_2, mturk_agent_1, mturk_agent_2])

        # Since we are using the regular MultiAgentDialogWorld, we do the following outside of the world instead.
        mturk_agent_ids = [mturk_agent_1.id, mturk_agent_2.id]
        all_agent_ids = [human_agent_1.id, human_agent_2.id] + mturk_agent_ids
        mturk_agent_1.mturk_agent_ids = mturk_agent_ids
        mturk_agent_1.all_agent_ids = all_agent_ids
        mturk_agent_2.mturk_agent_ids = mturk_agent_ids
        mturk_agent_2.all_agent_ids = all_agent_ids
        mturk_agent_1.create_hit()
        mturk_agent_2.create_hit()

        while not world.episode_done():
            world.parley()
        world.shutdown()

    mturk_manager = MTurkManager()
    mturk_manager.init_aws(opt=opt)
    results = Parallel(n_jobs=opt['num_hits'], backend='threading')(delayed(run_hit)(i, opt, mturk_manager) for i in range(1, opt['num_hits']+1))
    mturk_manager.review_hits()
    mturk_manager.shutdown()

if __name__ == '__main__':
    main()
