-- Copyright 2004-present Facebook. All Rights Reserved.
-- @lint-skip-luachecker

local cmdline = require('fbcode.deeplearning.projects.memnns.v1.cmd')
cmd = cmdline:new()

cmd:reset_default('modelClass',
    'fbcode.deeplearning.projects.memnns.v1.memnn_model')
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

cmd:reset_default('logEveryNSecs', 1)
cmd:reset_default('useCandidateLabels', false)
cmd:reset_default('initWeights', 0.01)
cmd:reset_default('learningRate', 0.001)
cmd:reset_default('useTimeFeatures', true)
cmd:reset_default('timeVariance', 3)
cmd:reset_default('rankLabelDocuments', false)
cmd:reset_default('embeddingDim', 20)
cmd:reset_default('maxHops', 2)
cmd:reset_default('memSize', 50)
cmd:reset_default('LTsharedWithResponse', false)

cmd:option('task', 1, 'Task number.')

opt = cmd:parse(arg)

local data_basename =
    '/mnt/vol/gfsai-east/ai-group/users/ahm/babisteps/qa' .. opt.task ..
    '/torch/'

cmd:reset_default('dictFile', data_basename .. 'dict.txt')
cmd:reset_default('trainData', data_basename .. 'train.txt.vecarray')
cmd:reset_default('validData', data_basename .. 'test.txt.vecarray')
cmd:reset_default('testData', data_basename .. 'test.txt.vecarray')

opt = cmd:parse(arg)

cmd:print(opt)

return opt
