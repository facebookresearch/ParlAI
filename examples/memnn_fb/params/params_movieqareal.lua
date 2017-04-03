-- Copyright 2004-present Facebook. All Rights Reserved.
-- @lint-skip-luachecker

local cmdline = require('fbcode.deeplearning.projects.memnns.v1.cmd')
cmd = cmdline:new()

cmd:reset_default('modelClass',
    'fbcode.deeplearning.projects.memnns.v1.bimemnn_model')
cmd:reset_default('dataClass',
    'fbcode.deeplearning.projects.memnns.v1.data')

opt = cmd:parse(arg, true)

local mlp = torch.reload(opt.modelClass)
mlp:add_cmdline_options(cmd)
local data = torch.reload(opt.dataClass)
data:add_cmdline_options(cmd)

cmd:reset_default('allowSaving', false)
cmd:reset_default('variableMemories', true)
cmd:reset_default('wordModel', 'bowTFIDF')
local base_path = '/mnt/vol/gfsai-east/ai-group/datasets/dialogue/movies/movieqa_real/torch/'
cmd:reset_default('dictFile', base_path .. 'dict.txt')
cmd:reset_default('memHashFile', base_path .. 'kb.txt.hash')
cmd:reset_default('trainData', base_path .. 'qa-train.txt.vecarray')
cmd:reset_default('validData', base_path .. 'qa-dev.txt.vecarray')
cmd:reset_default('testData', base_path .. 'qa-test.txt.vecarray')

opt = cmd:parse(arg)

cmd:print(opt)

return opt
