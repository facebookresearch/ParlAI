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
cmd:reset_default('useCandidateLabels', true)
cmd:reset_default('numNegSamples', 100)
cmd:reset_default('initWeights', 0.01)
cmd:reset_default('learningRate', 0.01)
cmd:reset_default('useTimeFeatures', true)
cmd:reset_default('timeVariance', 3)
cmd:reset_default('rankLabelDocuments', false)
cmd:reset_default('embeddingDim', 20)
cmd:reset_default('maxHops', 1)
cmd:reset_default('memSize', 50)
cmd:reset_default('LTsharedWithResponse', true)

opt = cmd:parse(arg)
cmd:reset_default('dictFile', '/tmp/dict.txt')

opt = cmd:parse(arg)

cmd:print(opt)

return opt
