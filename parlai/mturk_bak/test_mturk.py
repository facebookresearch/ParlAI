# Copyright 2004-present Facebook. All Rights Reserved.
from sample_bot import MTurkAgent
import eval_bot

eval_bot.create_hits(
	opt=None, 
	bot=MTurkAgent(opt=None), 
	num_hits=1, 
	message='opening message')