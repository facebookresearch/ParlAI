import time
from parlai.mturk.core.agents import MTurkAgent
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.core.worlds import MultiAgentDialogWorld

conversation_id = str(int(time.time())) # Use the same conversation_id for all mturk agents
task_name = 'multi_agent_dialog'
mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
mturk_agent_1 = MTurkAgent({'agent_id':'mturk_agent_1', 'task_name':task_name, 'is_sandbox':True, 'conversation_id':conversation_id, 'mturk_agent_ids':mturk_agent_ids})
mturk_agent_2 = MTurkAgent({'agent_id':'mturk_agent_2', 'task_name':task_name, 'is_sandbox':True, 'conversation_id':conversation_id, 'mturk_agent_ids':mturk_agent_ids})

human_agent_1 = LocalHumanAgent(opt=None)
human_agent_1.id = 'human_1'
human_agent_2 = LocalHumanAgent(opt=None)
human_agent_2.id = 'human_2'

world = MultiAgentDialogWorld({'task':task_name}, [human_agent_1, human_agent_2, mturk_agent_1, mturk_agent_2])

while not world.episode_done():
    world.parley()

world.shutdown()
