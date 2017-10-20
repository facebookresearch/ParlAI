# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""
Commands to communicate between the agent manager and individual clients. The
following describes the intended behavoir of each command

COMMAND_SEND_MESSAGE ...... / MTurk web client is expected to send new message
                            \  to the server. Allow the user to send a message
COMMAND_SHOW_DONE_BUTTON .. / MTurk web client should show the "DONE" button
                            \  with no special text
COMMAND_EXPIRE_HIT ........ / MTurk web client should expire the hit and update
                            |  the UI, additional text to display is in the
                            \  'inactive_text' param
COMMAND_SUBMIT_HIT ........ / MTurk web client should submit the HIT directly
COMMAND_CHANGE_CONVERSATION / MTurk web client should change conversations
                            | 'conversation_id' holds the new conversation_id
                            \ 'agent_id' holds the new display id for the agent
COMMAND_RESTORE_STATE ..... / MTurk web client should restore the state of a
                            |  disconnected conversation. The previously sent
                            |  messages are in the 'messages' param, and the
                            \  last sent command is in 'last_command'
COMMAND_INACTIVE_HIT ...... / MTurk web client should remove the done button
                            |  and text box and instead display the contents of
                            \  'inactive_text' param.
COMMAND_INACTIVE_DONE ..... / MTurk web client should show the "DONE" button
                            |  and display the contents of
                            \  'inactive_text' param.
"""
COMMAND_SEND_MESSAGE = 'COMMAND_SEND_MESSAGE'
COMMAND_SHOW_DONE_BUTTON = 'COMMAND_SHOW_DONE_BUTTON'
COMMAND_EXPIRE_HIT = 'COMMAND_EXPIRE_HIT'
COMMAND_SUBMIT_HIT = 'COMMAND_SUBMIT_HIT'
COMMAND_CHANGE_CONVERSATION = 'COMMAND_CHANGE_CONVERSATION'
COMMAND_RESTORE_STATE = 'COMMAND_RESTORE_STATE'
COMMAND_INACTIVE_HIT = 'COMMAND_INACTIVE_HIT'
COMMAND_INACTIVE_DONE = 'COMMAND_INACTIVE_DONE'

# Socket function names
SOCKET_OPEN_STRING = 'socket_open' # Event fires when a socket opens
SOCKET_DISCONNECT_STRING = 'disconnect' # Event fires when a socket disconnects
SOCKET_NEW_PACKET_STRING = 'new packet' # Event fires when packets arrive
SOCKET_ROUTE_PACKET_STRING = 'route packet' # Event to send outgoing packets
SOCKET_AGENT_ALIVE_STRING = 'agent alive' # Event to send alive packets

# Message types
MESSAGE_TYPE_MESSAGE = 'MESSAGE'
MESSAGE_TYPE_COMMAND = 'COMMAND'
