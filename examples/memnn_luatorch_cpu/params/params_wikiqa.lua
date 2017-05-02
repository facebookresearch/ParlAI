-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.
-- @lint-skip-luachecker

local cmdline = require('library.cmd')
cmd = cmdline:new()

cmd:reset_default('modelClass',
    'library.memnn_model')
cmd:reset_default('dataClass',
    'library.data')

opt = cmd:parse(arg, true)

local mlp = require(opt.modelClass)
mlp:add_cmdline_options(cmd)
local data = require(opt.dataClass)
data:add_cmdline_options(cmd)

cmd:reset_default('logEveryNSecs', 1)
cmd:reset_default('allowSaving', false)
cmd:reset_default('useCandidateLabels', true)
cmd:reset_default('numNegSamples', 100)
local base_path = '/mnt/vol/gfsai-east/ai-group/datasets/wikiqacorpus/' ..
                  'memnn-multians/torch/'
cmd:reset_default('dictFile', base_path .. 'dict.txt')
cmd:reset_default('trainData', base_path .. 'WikiQA-train.txt.vecarray')
cmd:reset_default('validData', base_path .. 'WikiQA-dev.txt.vecarray')
cmd:reset_default('testData', base_path .. 'WikiQA-test.txt.vecarray')

opt = cmd:parse(arg)

cmd:print(opt)

return opt
