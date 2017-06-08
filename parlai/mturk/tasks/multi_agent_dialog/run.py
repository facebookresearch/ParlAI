import os
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.agents import MTurkAgent
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.core.worlds import MultiAgentDialogWorld
from task_config import task_config

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

    mturk_agent_1_id = 'mturk_agent_1'
    mturk_agent_2_id = 'mturk_agent_2'
    human_agent_1_id = 'human_1'
    human_agent_2_id = 'human_2'

    # Create the MTurk agents
    opt.update(task_config)
    opt['conversation_id'] = str(int(time.time()))
    
    opt['mturk_agent_ids'] = [mturk_agent_1_id, mturk_agent_2_id]
    opt['all_agent_ids'] = [human_agent_1_id, human_agent_2_id, mturk_agent_1_id, mturk_agent_2_id]

    opt['agent_id'] = mturk_agent_1_id
    mturk_agent_1 = MTurkAgent(opt=opt)

    opt['agent_id'] = mturk_agent_2_id
    mturk_agent_2 = MTurkAgent(opt=opt)

    # Create the local human agents
    human_agent_1 = LocalHumanAgent(opt=None)
    human_agent_1.id = human_agent_1_id
    human_agent_2 = LocalHumanAgent(opt=None)
    human_agent_2.id = human_agent_2_id

    world = MultiAgentDialogWorld(opt=opt, agents=[human_agent_1, human_agent_2, mturk_agent_1, mturk_agent_2])

    while not world.episode_done():
        world.parley()

    world.shutdown()

if __name__ == '__main__':
    main()
