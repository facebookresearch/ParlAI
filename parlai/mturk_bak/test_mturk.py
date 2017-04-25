# Copyright 2004-present Facebook. All Rights Reserved.
import manage_hit

# Simple demo
# task_module_path_prefix = 'tasks.demo.'
# MTurkAgent = __import__(task_module_path_prefix+'agents', fromlist=['']).MTurkDemoAgent

# SQuAD data augmentation
task_module_path_prefix = 'tasks.squad_data_augmentation.'
MTurkAgent = __import__(task_module_path_prefix+'agents', fromlist=['']).MTurkSquadDataAugmentationAgent


task_config = __import__(task_module_path_prefix+'task_config', fromlist=['']).task_config
DataLoader = __import__(task_module_path_prefix+'data_loader', fromlist=['']).DataLoader

manage_hit.create_hits(
	opt=None, 
	task_config=task_config,
	data_loader=DataLoader(opt={'datapath': '../data/'}),
	bot=MTurkAgent(opt=None), 
	num_hits=1
)