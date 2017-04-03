-- Copyright 2004-present Facebook. All Rights Reserved.

-- Arguments:
-- 1) Name of file which contains options.

require('torch')
local zmq = require('lzmq')
local cjson = require('cjson')
local threads = require('threads')
threads.Threads.serialization('threads.sharedserialize')

-- make sure dependencies are set up
libdir = os.getenv('PARLAI_DOWNPATH') .. '/memnnlib'
if not os.rename(libdir, libdir) then
    print('could not find memnnlib, trying to clone it now')
    assert(os.execute('git clone git@github.com:facebook/MemNN.git ' .. libdir), 'error cloning into github.com/facebook/MemNN')
    print('executing: ' .. 'cd ' .. libdir .. '/KVMemnn/ && ./setup.sh')
    assert(os.execute('cd ' .. libdir .. '/KVMemnn/ && ./setup.sh'), 'error running ' .. libdir .. '/KVMemnn/setup.sh')
end
package.path =  libdir .. '/KVMemnn/?.lua' .. ';' .. package.path

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

local MemnnAgent = require('parlai.agents.memnn.memnn_luatorch_cpu.' ..
                           'memnn_agent_parsed')
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
                libdir = os.getenv('PARLAI_DOWNPATH') .. '/memnnlib'
                package.path =  libdir .. '/KVMemnn/?.lua' .. ';' .. package.path
                require('torch')
                require('tds')

                local zmq = require('lzmq')
                local cjson = require('cjson')

                if jobid ~= 1 then
                    jobopt['logEveryNSecs'] = nil
                end

                local MemnnAgent = require('parlai.agents.memnn.memnn_lua' ..
                                           'torch_cpu.memnn_agent_parsed')
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
