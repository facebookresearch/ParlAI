-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.
--[[
Takes in an observation table using the same format as usual for ParlAI, and
trains a memory network to produce the correct labels.

Observation table format:
{
    'text': 'What is the capital of Assyria?',
    'label': {
        1: 'Ninevah',
        2: 'Assur'
    },
    ...
}
--]]

require('torch')
local pl = require('pl.import_into')()
local sys = require('sys')
local tds = require('tds')
local threads = require('threads')
threads.Threads.serialization('threads.sharedserialize')

local memnn_agent = {}

function memnn_agent:__init(opt, shared)
    local agent = {}
    setmetatable(agent, {__index = memnn_agent})

    opt.dictFullLoading = true
    opt.allowSaving = false

    local modelClass = require(opt.modelClass)
    local datalib = require(opt.dataClass)

    if shared then
        agent.mlp = modelClass:clone_mlp(shared.mlp, opt)
        agent.data = datalib:create_data('', shared.data, opt, agent.mlp.dict)
        agent.metric_lock = threads.Mutex(shared.mutex_id)
        agent.metrics = shared.metrics
    else
        agent.mlp = modelClass:init_mlp(opt)
        agent.data = datalib:create_data('', nil, opt, agent.mlp.dict)
        agent.metric_lock = threads.Mutex()
        agent.metrics = tds.hash()
        agent.metrics.cnt = 0
        agent.metrics.cnt_since_log = 0
    end

    local parserlib = require(opt.parserClass)
    agent.parser = parserlib:new(opt)

    g_train_data = agent.data
    g_valid_data = agent.data

    agent.lastLogTime = sys.clock()
    agent.startTime = sys.clock()

    agent.logEveryNSecs = opt.logEveryNSecs

    agent.NULL = torch.Tensor(1, 2):fill(1)
    return agent
end

function memnn_agent:share()
    local shared = {}
    shared.mlp = self.mlp:get_shared()
    shared.data = self.data:get_shared()
    shared.mutex_id = self.metric_lock:id()
    shared.metrics = self.metrics

    return shared
end

-- update current state with given state/action table
function memnn_agent:act(reply)
    local metrics = self.metrics

    local mem = pl.stringx.split(reply.text, '\n')
    local query = mem[#mem]
    mem[#mem] = nil
    -- if there are multiple answers, pick one of them to train on
    local ans
    if reply.labels then
        ans = reply.labels[math.random(#reply.labels)]
    end

    local ex = self:_build_ex(query, ans, mem, reply.candidates, self.prev_ex)

    if reply.done then
        self.prev_ex = nil
    else
        self.prev_ex = ex
    end

    if not reply.labels then
        -- if you don't have labels, prepare to give an answer back in act()
        -- since this is validation or test phase
        local resp_vec, _, _, c_ind = self.mlp:predict(ex)

        local t = {}
        if reply.candidates then
            t.text = reply.candidates[c_ind]
        else
            t.text = self.mlp.dict:v2t(resp_vec)
        end

        return t
    else
        -- currently only logging during / for training examples
        assert(ex[2])
        self.metric_lock:lock()
        metrics.cnt = metrics.cnt + 1
        self.metric_lock:unlock()

        local update_loss = self.mlp:update(ex)
        if update_loss then
            self.metric_lock:lock()
            for k, v in pairs(update_loss) do
                metrics[k] = (metrics[k] or 0) + v
            end
            metrics.cnt_since_log = metrics.cnt_since_log + 1
            self.metric_lock:unlock()
        end


        if self.logEveryNSecs then
            local time = sys.clock()
            if time - self.lastLogTime > self.logEveryNSecs then
                self.lastLogTime = time
                self.metric_lock:lock()
                print(string.format(
                    '[ exs: %d | time: %ds | mean_rank: %.2f | '
                    .. 'resp_loss: %.2f | rank_loss: %.2f ]',
                    metrics.cnt,
                    time - self.startTime,
                    (metrics.mean_rank or -metrics.cnt_since_log)
                        / metrics.cnt_since_log,
                    (metrics.r or -metrics.cnt_since_log)
                        / metrics.cnt_since_log,
                    (metrics.rank_loss or -metrics.cnt_since_log)
                        / metrics.cnt_since_log
                ))
                self.metrics.cnt_since_log = 0
                self.metrics.mean_rank = nil
                self.metrics.r = nil
                self.metrics.rank_loss = nil
                self.metric_lock:unlock()
            end
        end
    end
    return {}
end

function memnn_agent:_prep(data)
    return self.data:addTFIDF(self.data:resolveUNK(data))
end

function memnn_agent:_build_ex(query, label, mem, cands, prev_ex)
    local x = self:_prep(self.parser:parse_test_time(query, self.mlp.dict))
    local y = self:_prep(self.parser:parse_test_time(label, self.mlp.dict))
    local ex = {}
    ex[1] = x
    ex[2] = y
    ex.memhx = {}
    ex.memyx = {}
    ex.cands = {}
    if cands then
        cands = pl.stringx.join('|', cands) -- convert cands back for parsing
        local c_ht = self.parser:parse_candidates(cands, self.mlp.dict)
        for i = 1, #c_ht do
            table.insert(ex.cands, self:_prep(c_ht[i]))
        end
    end
    if self.data.hash then
        local c1, c2 = self.data.hash:get_candidate_set(ex[1])
        for i = 1, #c1 do
            c1[i] = self:_prep(c1[i])
            c2[i] = self:_prep(c2[i], true)
        end
        if #c2 > 0 and opt.maxHashSizeSortedByTFIDF > 0 then
            c1, c2 = self.data.hash:sortByTFIDF(x, c1, c2)
        end
        ex.memhx = c1
        ex.memhy = c2
    end


    if not prev_ex then
        ex.memx = {}
    else
        ex.memx = prev_ex.memx
        table.insert(ex.memx, prev_ex[1])

        ex.memy = prev_ex.memy or {}
        for i = #ex.memy, #ex.memx - 2 do table.insert(ex.memy, self.NULL) end
        table.insert(ex.memy, prev_ex[2])
    end
    if mem then
        for _, m in ipairs(mem) do
            table.insert(ex.memx, self.parser:parse_test_time(m, self.mlp.dict))
        end
        if ex.memy then
            for i = #ex.memy, #ex.memx - 1 do
                table.insert(ex.memy, self.NULL)
            end
        end
    end
    return ex
end

return memnn_agent
