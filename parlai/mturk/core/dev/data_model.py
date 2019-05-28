#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Commands to communicate between the agent manager and individual clients. The
following describes the intended behavoir of each command

COMMAND_SEND_MESSAGE ...... / MTurk web client is expected to send new message
                            \\ to the server. Allow the user to send a message
COMMAND_SUBMIT_HIT ........ / MTurk web client should submit the HIT directly
"""
COMMAND_SEND_MESSAGE = 'COMMAND_SEND_MESSAGE'
COMMAND_SUBMIT_HIT = 'COMMAND_SUBMIT_HIT'

# Socket function names / packet types
# TODO document
WORLD_MESSAGE = 'world message'  # Message from world to agent
AGENT_MESSAGE = 'agent message'  # Message from agent to world
WORLD_PING = 'world ping'  # Ping from the world for this server uptime
SERVER_PONG = 'server pong'  # pong to confirm uptime
MESSAGE_BATCH = 'message batch'  # packet containing batch of messages
AGENT_DISCONNECT = 'agent disconnect'  # Notes an agent disconnecting
SNS_MESSAGE = 'sns message'   # packet from an SNS message
STATIC_MESSAGE = 'static message'  # packet from static done POST
AGENT_STATE_CHANGE = 'agent state change'  # state change from parlai
AGENT_ALIVE = 'agent alive'  # packet from an agent alive event

# Message types
MESSAGE_TYPE_ACT = 'MESSAGE'
MESSAGE_TYPE_COMMAND = 'COMMAND'
