import os
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.agents import MTurkAgent
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
    global run_hit
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.getcwd())

    opt['mturk_agent_1_id'] = 'mturk_agent_1'
    opt['mturk_agent_2_id'] = 'mturk_agent_2'
    opt['human_agent_1_id'] = 'human_1'
    opt['human_agent_2_id'] = 'human_2'

    opt.update(task_config)
    opt['mturk_agent_ids'] = [opt['mturk_agent_1_id'], opt['mturk_agent_2_id']]
    opt['all_agent_ids'] = [opt['human_agent_1_id'], opt['human_agent_2_id']] + opt['mturk_agent_ids']
    opt['run_id'] = str(int(time.time()))

    def run_hit(i, opt):
        opt['conversation_id'] = str(i)

        opt['agent_id'] = opt['mturk_agent_1_id']
        mturk_agent_1 = MTurkAgent(opt=opt)
        opt['agent_id'] = opt['mturk_agent_2_id']
        mturk_agent_2 = MTurkAgent(opt=opt)

        # Create the local human agents
        human_agent_1 = LocalHumanAgent(opt=None)
        human_agent_1.id = opt['human_agent_1_id']
        human_agent_2 = LocalHumanAgent(opt=None)
        human_agent_2.id = opt['human_agent_2_id']

        world = MultiAgentDialogWorld(opt=opt, agents=[human_agent_1, human_agent_2, mturk_agent_1, mturk_agent_2])
        while not world.episode_done():
            world.parley()
        world.shutdown()

    MTurkAgent.init_aws(opt)
    results = Parallel(n_jobs=opt['num_hits'], backend='threading')(delayed(run_hit)(i, copy.deepcopy(opt)) for i in range(1, opt['num_hits']+1))
    MTurkAgent.review_hits(opt=opt)

if __name__ == '__main__':
    main()
