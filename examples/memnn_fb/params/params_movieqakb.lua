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
local base_path = '/mnt/vol/gfsai-east/ai-group/datasets/dialogue/movies/movieqa_kb/torch/'
cmd:reset_default('dictFile', base_path .. 'dict-jase-bimemnn.txt')
cmd:reset_default('memHashFile', base_path .. 'movieqa_kb-jase-bimemnn.txt.hash')
cmd:reset_default('trainData', base_path .. 'wiki-entities_qa_train-jase-bimemnn.txt.vecarray')
cmd:reset_default('validData', base_path .. 'wiki-entities_qa_dev-jase-bimemnn.txt.vecarray')
cmd:reset_default('testData', base_path .. 'wiki-entities_qa_test-jase-bimemnn.txt.vecarray')

opt = cmd:parse(arg)

cmd:print(opt)

return opt
