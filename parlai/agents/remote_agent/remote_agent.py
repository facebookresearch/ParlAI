# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.agents import Agent, create_agent_from_shared
import copy
import numpy as np
import json
import subprocess
import zmq


class RemoteAgentAgent(Agent):
    """Agent which connects over ZMQ to a paired agent. The other agent is
    launched using the command line options set via `add_cmdline_args`."""

    @staticmethod
    def add_cmdline_args(argparser):
        argparser.add_argument(
            '--port', default=5555,
            help='first port to connect to for remote agents')
        argparser.add_argument(
            '--remote-address', default='localhost',
            help='address to connect to')
        argparser.add_argument(
            '--remote-host', action='store_true',
            help='whether or not this connection is the host or the client')
        argparser.add_argument(
            '--remote-cmd',
            help='command to launch paired agent, if applicable')
        argparser.add_argument(
            '--remote-args',
            help='optional arguments to pass to paired agent')

    def __init__(self, opt, shared=None):
        """Runs subprocess command to set up remote partner.
        Only run the subprocess command once: if using multiple threads, tell
        the partner how many paired agents to set up so that they can manage
        the multithreading effectively in their environment. (We don't run
        subprocess.Popen for each thread.)
        """
        if shared and 'port' in shared:
            # for multithreading
            self.port = shared['port']
            self.address = shared['address']
            self.opt = copy.deepcopy(shared['opt'])
        else:
            self.opt = copy.deepcopy(opt)
            if 'port' in opt:
                self.port = opt['port']
                self.address = opt['remote_address']
                self.socket_type = zmq.REP if opt['remote_host'] else zmq.REQ
            else:
                raise RuntimeError('You need to run RemoteAgent.' +
                                   'add_cmdline_args(argparser) before ' +
                                   'calling this class to set up options.')
            if opt.get('remote_cmd'):
                # if available, command to launch partner instance, passing on
                # some shared parameters from ParlAI
                # useful especially if "remote" agent is running locally, e.g.
                # in a different language than python
                self.process = subprocess.Popen(
                    '{cmd} {port} {numthreads} {args}'.format(
                        cmd=opt['remote_cmd'], port=opt['port'],
                        numthreads=opt['numthreads'],
                        args=opt.get('remote_args', '')
                    ).split()
                )
        self.connect()
        super().__init__(opt, shared)

    def connect(self):
        """Connect to ZMQ socket as client. Requires package zmq."""
        context = zmq.Context()
        self.socket = context.socket(self.socket_type)
        self.socket.setsockopt(zmq.LINGER, 1)
        host = 'tcp://{}:{}'.format(self.address, self.port)
        if self.socket_type == zmq.REP:
            self.socket.bind(host)
        else:
            self.socket.connect(host)
        print('python thread connected to ' + host)

    def act(self):
        """Send message to paired agent listening over zmq."""
        if self.observation is not None:
            if 'image' in self.observation:
                # can't json serialize images
                self.observation.pop('image', None)
            text = json.dumps(self.observation)
            self.socket.send_unicode(text)
        reply = self.socket.recv_unicode()
        return json.loads(reply)

    def share(self):
        """Increments port to use when using remote agents in Hogwild mode."""
        if not hasattr(self, 'lastport'):
            self.lastport = self.port
        shared = {}
        shared['port'] = self.lastport + 1
        shared['class'] = type(self)
        shared['opt'] = self.opt
        self.lastport += 1
        return shared

    def shutdown(self):
        """Shut down paired listener with <END> signal."""
        if hasattr(self, 'socket'):
            try:
                self.socket.send_unicode('<END>', zmq.NOBLOCK)
            except zmq.error.ZMQError:
                # may need to listen first
                try:
                    self.socket.recv_unicode(zmq.NOBLOCK)
                    self.socket.send_unicode('<END>', zmq.NOBLOCK)
                except zmq.error.ZMQError:
                    # paired process is probably dead already
                    pass
        if hasattr(self, 'process'):
            # try to let the subprocess clean up, but don't wait too long
            try:
                self.process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()


class ParsedRemoteAgent(RemoteAgentAgent):
    """Same as the regular remote agent, except that this agent converts all
    text into vectors using its dictionary before sending them.
    """

    def __init__(self, opt, shared=None):
        if 'dictionary_agent' in shared:
            # use this first--maybe be overriding an original dictionary
            self.dict = create_agent_from_shared(shared['dictionary_agent'])
        elif 'dictionary' in shared:
            # otherwise use this dictionary
            self.dict = shared['dictionary']
        else:
            raise RuntimeError('ParsedRemoteAgent needs a dictionary to parse' +
                               ' text with--pass in a dictionary using shared' +
                               '["dictionary"] or pass in the arguments to ' +
                               'instantiate one using shared["dictionary_args' +
                               '"] = (class, options, shared).')
        super().__init__(opt, shared)

    def act(self):
        parsed = {}
        for k, v in self.observation.items():
            if type(v) == str:
                # We split on newlines because we don't treat them as charactes
                # in the default dictionary but our receiving agent might want
                # to know where the newlines are in the text block.
                parsed[k] = self.parse(v, split_lines=True)
            else:
                # not a string, see if it's an iterable of strings
                try:
                    # if there are newlines in the label, it's part of the label
                    parsed[k] = [self.parse(s) for s in v]
                except TypeError:
                    # oops, it's not. just pass it on.
                    parsed[k] = v
        super().observe(parsed)
        reply = super().act()
        unparsed = {}
        for k, v in reply.items():
            # TODO(ahm): this fails if remote agent sends anything other than
            # vectors, which means that pretty much only .text will work
            unparsed[k] = self.parse(v)
        return unparsed

    def parse(self, s, split_lines=False):
        """Returns a parsed (list of indices) version of a string s.
        Optionally return list of vectors for each line in the string in case
        you need to know where those are.
        """
        if split_lines:
            return [self.dict.parse(line, vec_type=list)
                    for line in s.split('\n')]
        else:
            return self.dict.parse(s, vec_type=list)

    def share(self):
        shared = super().share()
        shared['dictionary_agent'] = self.dict.share()
        return shared
