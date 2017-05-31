# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.params import ParlaiParser
from core import manage_hit

argparser = ParlaiParser(False, False)
argparser.add_parlai_data_path()
argparser.add_mturk_args()

opt = argparser.parse_args()

task_module_name = 'parlai.mturk.tasks.' + opt['task']
Agent = __import__(task_module_name+'.agents', fromlist=['']).default_agent_class
task_config = __import__(task_module_name+'.task_config', fromlist=['']).task_config

print("Creating HIT tasks for "+task_module_name+" ...")

manage_hit.create_hits(
	opt=opt,
	task_config=task_config,
	task_module_name=task_module_name,
	bot=Agent(opt=opt),
)