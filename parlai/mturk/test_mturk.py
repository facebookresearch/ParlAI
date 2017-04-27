# Copyright 2004-present Facebook. All Rights Reserved.
import manage_hit

# Simple demo
# task_module_path_prefix = 'tasks.demo'
# MTurkAgent = __import__(task_module_path_prefix+'.agents', fromlist=['']).MTurkDemoAgent

# SQuAD data collection
task_module_path_prefix = 'tasks.squad_data_collection'
MTurkAgent = __import__(task_module_path_prefix+'.agents', fromlist=['']).MTurkQADataCollectionAgent

# SQuAD eval
# task_module_path_prefix = 'tasks.squad_eval'
# MTurkAgent = __import__(task_module_path_prefix+'.agents', fromlist=['']).MTurkSquadEvalAgent


task_config = __import__(task_module_path_prefix+'.task_config', fromlist=['']).task_config
DataLoader = __import__(task_module_path_prefix+'.data_loader', fromlist=['']).DataLoader

print("Creating HIT tasks for "+task_module_path_prefix+" ...")
manage_hit.create_hits(
	task_config=task_config,
	data_loader=DataLoader(opt={'datapath': '../data/'}),
	bot=MTurkAgent(opt={}), 
	num_hits=2,
	is_sandbox=True,
	chat_page_only=False
)