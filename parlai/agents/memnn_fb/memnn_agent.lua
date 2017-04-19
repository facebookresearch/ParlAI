-- Copyright 2004-present Facebook. All Rights Reserved.

-- import open source memnn agent since they're the same--they just use a
-- different library
local memnn_agent = require('fbcode.deeplearning.projects.parlai.parlai.' ..
                            'agents.memnn_luatorch_cpu.memnn_agent')
return memnn_agent
