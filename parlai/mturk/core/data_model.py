# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Commands
COMMAND_SEND_MESSAGE = 'COMMAND_SEND_MESSAGE' # MTurk web client is expected to send a new message to server
COMMAND_SHOW_DONE_BUTTON = 'COMMAND_SHOW_DONE_BUTTON' # MTurk web client should show the "DONE" button
COMMAND_EXPIRE_HIT = 'COMMAND_EXPIRE_HIT' # MTurk web client should show "HIT is expired"
COMMAND_SUBMIT_HIT = 'COMMAND_SUBMIT_HIT' # MTurk web client should submit the HIT directly
COMMAND_CHANGE_CONVERSATION = 'COMMAND_CHANGE_CONVERSATION' # MTurk web client should change to new conversation
COMMAND_RESTORE_STATE = 'COMMAND_RESTORE_STATE'
COMMAND_DISCONNECT_PARTNER = 'COMMAND_DISCONNECT_PARTNER'
COMMAND_INACTIVE_HIT = 'COMMAND_INACTIVE_HIT'
COMMAND_INACTIVE_DONE = 'COMMAND_INACTIVE_DONE'

# Socket function names
SOCKET_OPEN_STRING = 'socket_open'
SOCKET_DISCONNECT_STRING = 'disconnect'
SOCKET_NEW_PACKET_STRING = 'new packet'
SOCKET_ROUTE_PACKET_STRING = 'route packet'
SOCKET_AGENT_ALIVE_STRING = 'agent alive'

# Message types
MESSAGE_TYPE_MESSAGE = 'MESSAGE'
MESSAGE_TYPE_COMMAND = 'COMMAND'
