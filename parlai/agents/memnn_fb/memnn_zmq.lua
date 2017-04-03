-- Copyright 2004-present Facebook. All Rights Reserved.
--[[
This connects to local ports over ZMQ, parsing incoming json into a table and
forwarding that to the memnn_agent. This agent assumes that incoming data
is formatted as text.

Sets up as many threads as needed, starting with the port number and counting
up from there. Note: actually connects to n+1 ports, since the first port is
often used by a single-threaded validation thread and the remaining n by the
training threads.
--]]

require('fbtorch')
local zmq = require('lzmq')
local cjson = require('cjson')
local threads = require('threads')
threads.Threads.serialization('threads.sharedserialize')

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
        local cmdline = require('fbcode.deeplearning.projects.memnns.v1.cmd')
        cmd = cmdline:new()
        opt = cmd:parse_from_modelfile_and_override_with_args(arg)
        cmd:print(opt)
        print("[from loaded options]")
    elseif optFile == 'nil' then
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

        opt = cmd:parse(arg)

        cmd:print(opt)
        print("[default options]")
    else
        opt = torch.reload(optFile)
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
opt.parserClass = 'fbcode.deeplearning.projects.memnns.v1.parse'

local MemnnAgent = require('fbcode.deeplearning.projects.parlai.parlai.' ..
                           'agents.memnn_fb.memnn_agent')
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
                os.execute('sleep ' .. tostring(0.005 * jobid))
                require('fbtorch')
                require('tds')

                local zmq = require('lzmq')
                local cjson = require('cjson')

                if jobid ~= 1 then
                    jobopt['logEveryNSecs'] = nil
                end
                local MemnnAgent = require('fbcode.deeplearning.projects.' ..
                                           'parlai.parlai.agents.memnn_fb.' ..
                                           'memnn_agent')
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
