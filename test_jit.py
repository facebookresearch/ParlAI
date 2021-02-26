#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.jit

from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser

parser = ParlaiParser(add_parlai_args=True, add_model_args=True)
args_ = f"""\
--model-file zoo:blender/blender_90M/model \
--no-cuda \
"""
opt = parser.parse_args(args_.split())
agent = create_agent(opt, requireModelExists=True)
# Using create_agent() instead of create_agent_from_model_file() because I couldn't get
# --no-cuda to be recognized with the latter
# get the tokenization
obs = agent.observe({'text': 'hello world', 'episode_done': True})
batch = agent.batchify([obs])
tokens = batch.text_vec

# Run the original greedy search and trace the result
result = agent.model.jit_greedy_search(tokens)
# print(result)
print(agent._v2t(result[0].tolist()))

# Trace and save the module
traced_module = torch.jit.trace_module(agent.model, {'jit_greedy_search': tokens})
print('Finished tracing.')
traced_module.save('_traced_blender_90M.pt')

# Run greedy search on the traced module
traced_result = traced_module.jit_greedy_search(tokens)
print(agent._v2t(traced_result[0].tolist()))
