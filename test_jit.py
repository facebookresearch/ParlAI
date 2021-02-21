#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.jit

from parlai.core.agents import create_agent_from_model_file

agent = create_agent_from_model_file("zoo:blender/blender_90M/model")
# get the tokenization
obs = agent.observe({'text': 'hello world', 'episode_done': True})
batch = agent.batchify([obs])
tokens = batch.text_vec

result = agent.model.jit_greedy_search(tokens)
print(result)
print(agent._v2t(result[0].tolist()))

trace = torch.jit.trace_module(agent.model, {'jit_greedy_search': tokens})
print(trace)
