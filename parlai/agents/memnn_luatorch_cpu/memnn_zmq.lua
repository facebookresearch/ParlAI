-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.
--[[
This connects to local ports over ZMQ, parsing incoming json into a table and
forwarding that to the memnn_agent. This agent assumes that incoming data
is formatted as text.

Sets up as many threads as needed, starting with the port number and counting
up from there. Note: actually connects to n+1 ports, since the first port is
often used by a single-threaded validation thread and the remaining n by the
training threads.
--]]

-- make sure dependencies are set up
local libdir = os.getenv('PARLAI_DOWNPATH') .. '/memnnlib'
if not os.rename(libdir, libdir) then
    print('could not find memnnlib, trying to clone it now')
    assert(os.execute('git clone https://github.com/facebook/MemNN.git ' .. libdir),
           'error cloning into github.com/facebook/MemNN')
    assert(os.execute('cd ' .. libdir .. '/KVmemnn/ && ./setup.sh'),
           'error running ' .. libdir .. '/KVmemnn/setup.sh')
end
package.path =  libdir .. '/KVmemnn/?.lua' .. ';' .. package.path

require('torch')
local zmq = require('lzmq')
local cjson = require('cjson')
local threads = require('threads')
threads.Threads.serialization('threads.sharedserialize')

-- read args
local port = assert(arg[1], 'need port')
local numThreads = assert(tonumber(arg[2]), 'need numThreads')
local optFile = assert(arg[3], 'need optfile')

local opt
if optFile ~= nil then
    -- Kill all the args apart from the ones after the first
    -- to skip the options class.
    local args = {}
    for i = 5, #arg do
        args[#args + 1] = arg[i]
    end
    arg = args

    if optFile == 'model' then
        -- Load from model options instead.
        local cmdline = require('library.cmd')
        cmd = cmdline:new()
        opt = cmd:parse_from_modelfile_and_override_with_args(arg)
        cmd:print(opt)
        print("[from loaded options]")
    elseif optFile == 'nil' then
        local cmdline = require('library.cmd')
        cmd = cmdline:new()

        cmd:reset_default('modelClass', 'library.kvmemnn_model')
        cmd:reset_default('dataClass', 'library.data')

        opt = cmd:parse(arg, true)

        local mlp = require(opt.modelClass)
        mlp:add_cmdline_options(cmd)
        local data = require(opt.dataClass)
        data:add_cmdline_options(cmd)

        opt = cmd:parse(arg)

        cmd:print(opt)
        print("[default options]")
    else
        opt = dofile(optFile)
    end
else
    error('missing options file')
end

-- creates a zmq context then listens for queries and returns agent's replies
local function dojob(zmq, cjson, port, agent)
    local context = zmq.context()
    local responder, err = context:socket{zmq.REP, bind='tcp://*:' .. port}
    assert(responder, tostring(err))
    print('lua thread bound to '  .. 'tcp://*:' .. port)

    while true do
        local buffer = responder:recv()
        if not buffer or buffer == '<END>' then break end -- no more messages
        local t = cjson.decode(buffer)
        for k, v in pairs(t) do if v == cjson.null then t[k] = nil end end
        local reply = agent:act(t)
        local json = cjson.encode(reply)
        responder:send(json)
    end
    responder:send('<ACK> from ' .. (port - 5555))
    responder:close()
end

-- setup main agent
opt.parserClass = 'library.parse'

package.path = os.getenv('PARLAI_HOME') .. '/?.lua;' .. package.path
local MemnnAgent = require('parlai.agents.memnn_luatorch_cpu' ..
                           '.memnn_agent')
local agent = MemnnAgent:__init(opt)

if numThreads == 1 then
    -- if just one thread--go
    dojob(zmq, cjson, port, agent)
else
    -- otherwise create multiple threads, each with own agent and zmq port
    local shared = agent:share()

    local pool = threads.Threads(numThreads + 1)

    for j = 0, numThreads do
        pool:addjob(
            function(jobid, jobopt)
                -- bug in torch threading can cause sometimes deadlock on
                -- concurrent imports, so take a quick nap before starting...
                os.execute('sleep ' .. tostring(0.05 * jobid))
                local libdir = os.getenv('PARLAI_DOWNPATH') .. '/memnnlib'
                package.path = libdir .. '/KVmemnn/?.lua' .. ';' .. package.path
                require('torch')
                require('tds')

                local zmq = require('lzmq')
                local cjson = require('cjson')

                if jobid ~= 1 then
                    jobopt['logEveryNSecs'] = nil
                end

                package.path =
                    os.getenv('PARLAI_HOME') .. '/?.lua;' .. package.path
                local MemnnAgent = require('parlai.agents.memnn_luatorch_cpu' ..
                                           '.memnn_agent')
                local agent = MemnnAgent:__init(jobopt, shared)
                local job_port = port + jobid

                dojob(zmq, cjson, job_port, agent)
            end,
            function() end,
            j, opt
        )
    end
    pool:synchronize()
    pool:terminate()
end
