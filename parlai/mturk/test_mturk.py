# Copyright 2004-present Facebook. All Rights Reserved.
from parlai.core.params import ParlaiParser
from core import manage_hit

# QA data collection
task_module_path_prefix = 'tasks.qa_data_collection'
Agent = __import__(task_module_path_prefix+'.agents', fromlist=['']).QADataCollectionAgent

# Model evaluator
# task_module_path_prefix = 'tasks.model_evaluator'
# Agent = __import__(task_module_path_prefix+'.agents', fromlist=['']).ModelEvaluatorAgent


task_config = __import__(task_module_path_prefix+'.task_config', fromlist=['']).task_config

print("Creating HIT tasks for "+task_module_path_prefix+" ...")

argparser = ParlaiParser(False, False)
argparser.add_parlai_data_path()

manage_hit.create_hits(
	task_config=task_config,
	bot=Agent(opt=argparser.parse_args()), 
	num_hits=2,
	is_sandbox=True,
	chat_page_only=False,
	verbose=True
)