# Copyright 2004-present Facebook. All Rights Reserved.
import manage_hit

# QA data collection
# task_module_path_prefix = 'tasks.qa_data_collection'
# Agent = __import__(task_module_path_prefix+'.agents', fromlist=['']).QADataCollectionAgent

# Data evaluator
task_module_path_prefix = 'tasks.data_evaluator'
Agent = __import__(task_module_path_prefix+'.agents', fromlist=['']).DataEvaluatorAgent


task_config = __import__(task_module_path_prefix+'.task_config', fromlist=['']).task_config

print("Creating HIT tasks for "+task_module_path_prefix+" ...")
manage_hit.create_hits(
	task_config=task_config,
	bot=Agent(opt={'datapath': '../data'}), 
	num_hits=2,
	is_sandbox=True,
	chat_page_only=False,
	verbose=True
)