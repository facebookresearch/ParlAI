-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.
--[[
This file has the same functionality as memnn_agent, except that it expects
(and returns) a zero-indexed vector (as a table) of word indices into the
dictionary instead of strings.
See memnn_agent for more details.

Parsed zmq message table format:
{
    'text': {1: word_idx1, 2: word_idx2, 3: word_idx3},
    'label': {
        1: {1: ans1_idx},
        2: {1: ans2_idx}
    },
    ...
}
--]]

require('torch')
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

    local query
    local mem = reply.text
    if mem then
        query = mem[#mem]
        mem[#mem] = nil
    else
        return {}
    end
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
            t.text = torch.totable(resp_vec:add(-1))
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
    data:add(1)
    return self.data:addTFIDF(self.data:resolveUNK(data))
end

function memnn_agent:_build_ex(query, label, mem, cands, prev_ex)
    local x = self:_prep(torch.Tensor(query))
    local y = label and self:_prep(torch.Tensor(label)) or self.NULL
    local ex = {}
    ex[1] = x
    ex[2] = y
    ex.memhx = {}
    ex.memyx = {}
    ex.cands = {}
    if cands then
        for _, c in pairs(cands) do
            table.insert(ex.cands, self:_prep(torch.Tensor(c)))
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
            table.insert(ex.memx, self:_prep(torch.Tensor(m)))
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
