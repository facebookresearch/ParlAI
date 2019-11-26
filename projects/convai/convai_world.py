#!/usr/bin/env python3

# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.worlds import World
from parlai.utils.misc import display_messages
from parlai.core.agents import create_agent_from_shared

import requests
import json
import time


class ConvAIWorld(World):
    """
    ConvAIWorld provides conversations with participants in the convai.io competition.

    This world takes in exactly one agent which will converse with a partner over a
    remote connection. For each remote conversation being maintained by this world, a
    copy of the original agent will be instantiated from the original agent's `share()`
    method.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        convai = argparser.add_argument_group('ConvAI Arguments')
        convai.add_argument(
            '-bi',
            '--bot-id',
            required=True,
            help='Id of local bot used to communicate with RouterBot',
        )
        convai.add_argument(
            '-bc',
            '--bot-capacity',
            type=int,
            default=-1,
            help='The maximum number of open dialogs. Use -1 '
            'for unlimited number of open dialogs',
        )
        convai.add_argument(
            '-rbu', '--router-bot-url', required=True, help='Url of RouterBot'
        )
        convai.add_argument(
            '-rbpd',
            '--router-bot-pull-delay',
            type=int,
            default=1,
            help='Delay before new request to RouterBot: minimum 1 sec',
        )
        convai.add_argument(
            '-m',
            '--max-pull-delay',
            type=int,
            default=600,
            help='Maximum delay for new requests if case of server ' 'unavailability',
        )

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, shared)

        if len(agents) != 1:
            raise RuntimeError('Need agent to talk to.')
        self.agent = agents[0]
        self.shared = agents[0].share()
        # Current chat id
        self.curr_chatID = None
        # All active and finished (but not cleared yet) chats
        self.chats = {}
        # Finished chats
        self.finished_chats = set()
        # Pairs of exchanges between remote and local agents (for printing)
        self.last_exchanges = dict()
        # Pool of messages from RouterBot
        self.messages = []
        # Url of RouterBot
        self.router_bot_url = opt['router_bot_url']
        # Delay before new request to RouterBot: minimum 1 sec
        self.router_bot_pull_delay = opt['router_bot_pull_delay']
        if self.router_bot_pull_delay < 1:
            self.router_bot_pull_delay = 1
        # Minimal pull delay is equal to initial value of router_bot_pull_delay
        self.minimum_pull_delay = self.router_bot_pull_delay
        # Maximum delay couldn't be smaller than minimum_pull_delay
        self.maximum_pull_delay = opt['max_pull_delay']
        if self.maximum_pull_delay < self.minimum_pull_delay:
            self.maximum_pull_delay = self.minimum_pull_delay
        # Id of local bot used to communicate with RouterBot
        self.bot_id = opt['bot_id']
        # The maximum number of open dialogs.
        # Use -1 for unlimited number of open dialogs
        self.bot_capacity = opt['bot_capacity']
        # RouterBot url with current bot id
        self.bot_url = self.router_bot_url + self.bot_id

    def _get_updates(self):
        """
        Make HTTP request to Router Bot for new messages Expecting server response to be
        like {'ok': True, "result": [...]}

        :return: list of new messages received since last request
        """
        res = requests.get(self.bot_url + '/getUpdates')
        if res.status_code != 200:
            print(res.text)
            self._increase_delay()
            return {'ok': False, "result": []}
        elif self.router_bot_pull_delay > self.minimum_pull_delay:
            self.router_bot_pull_delay = self.minimum_pull_delay
        return res.json()

    def _increase_delay(self):
        if self.router_bot_pull_delay < self.maximum_pull_delay:
            self.router_bot_pull_delay *= 2
            if self.router_bot_pull_delay > self.maximum_pull_delay:
                self.router_bot_pull_delay = self.maximum_pull_delay
            print('Warning! Increasing pull delay to %d', self.router_bot_pull_delay)

    def _send_message(self, observation, chatID):
        """
        Make HTTP request to Router Bot to post new message.

        :param observation: message that will be sent to server
        :param chatID: id of chat
        :return: None
        """
        if self._is_end_of_conversation(observation['text']):
            data = {
                'text': '/end',
                'evaluation': {'quality': 0, 'breadth': 0, 'engagement': 0},
            }
        else:
            data = {'text': observation['text'], 'evaluation': 0}
        message = {'chat_id': chatID, 'text': json.dumps(data)}

        headers = {'Content-Type': 'application/json'}

        res = requests.post(
            self.bot_url + '/sendMessage', json=message, headers=headers
        )
        if res.status_code != 200:
            print(res.text)
            res.raise_for_status()

    @staticmethod
    def _is_begin_of_conversation(message):
        return message.startswith('/start')

    @staticmethod
    def _is_end_of_conversation(message):
        return message.startswith('/end')

    @staticmethod
    def _is_skip_response(message):
        return message == ''

    @staticmethod
    def _get_chat_id(message):
        return message['message']['chat']['id']

    @staticmethod
    def _get_message_text(message):
        return message['message']['text']

    @staticmethod
    def _strip_start_message(message):
        lines = message.split('\n')[1:]
        lines = ['your persona: ' + line for line in lines]
        return '\n'.join(lines)

    def _init_chat(self, chatID):
        """
        Create new chat for new dialog. Sets up a new instantiation of the agent so that
        each chat has its own local state.

        :param chatID: chat id
        :return: new instance of your local agent
        """
        agent_info = self.shared

        # Add refs to current world instance and chat id to agent 'opt' parameter
        if 'opt' not in agent_info.keys() or agent_info['opt'] is None:
            agent_info['opt'] = {}
        agent_info['opt']['convai_chatID'] = chatID

        local_agent = create_agent_from_shared(agent_info)
        self.chats[chatID] = local_agent
        return self.chats[chatID]

    def cleanup_finished_chat(self, chatID):
        """
        Shutdown specified chat.

        :param chatID: chat id
        :return: None
        """
        if chatID in self.finished_chats:
            self.chats.pop(chatID).shutdown()
            self.finished_chats.remove(chatID)

    def cleanup_finished_chats(self):
        """
        Shutdown all finished chats.

        :return: None
        """
        for chatID in self.finished_chats.copy():
            self.cleanup_finished_chat(chatID)

    def pull_new_messages(self):
        """
        Requests the server for new messages and processes every message. If a message
        starts with '/start' string then a new chat will be created and the message will
        be added to stack. If a message has the same chat id as already existing chat
        then it will be added to message stack for this chat. Any other messages will be
        ignored. If after processing all messages message stack is still empty then new
        request to server will be performed.

        :return: None
        """
        print('Waiting for new messages from server...', flush=True)
        while True:
            time.sleep(self.router_bot_pull_delay)
            msgs = self._get_updates()
            if len(msgs["result"]) > 0:
                for msg in msgs["result"]:
                    print('\nProceed message: %s' % msg)
                    text = self._get_message_text(msg)
                    chatID = self._get_chat_id(msg)

                    if self.chats.get(chatID, None) is not None:
                        print('Message was recognized as part of chat #%s' % chatID)
                        self.messages.append((chatID, text))
                    elif self._is_begin_of_conversation(text):
                        print(
                            'Message was recognised as start of new chat #%s' % chatID
                        )
                        if self.bot_capacity == -1 or 0 <= self.bot_capacity > (
                            len(self.chats) - len(self.finished_chats)
                        ):
                            self._init_chat(chatID)
                            text = self._strip_start_message(text)
                            self.messages.append((chatID, text))
                            print(
                                'New world and agents for chat #%s are created.'
                                % chatID
                            )
                        else:
                            print(
                                'Cannot start new chat #%s due to bot capacity'
                                'limit reached.' % chatID
                            )
                    else:
                        print(
                            'Message was not recognized as part of any chat.'
                            'Message skipped.'
                        )
                if len(self.messages) > 0:
                    break
                else:
                    print('Waiting for new messages from server...', flush=True)

    def parley(self):
        """
        Pops next message from stack, gets corresponding chat, agents, world and
        performs communication between agents. Result of communication will be send to
        server. If message stack is empty then server will be requested for new
        messages.

        :return: None
        """
        print('Try to cleanup finished chat before new parley.')
        self.cleanup_finished_chats()

        if len(self.messages) == 0:
            print('Message stack is empty. Try to request new messages from server.')
            self.pull_new_messages()

        print('Pop next message from stack')

        (chatID, text) = self.messages.pop(0)
        episode_done = self._is_end_of_conversation(text)
        local_agent = self.chats.get(chatID, None)

        if local_agent is not None:
            self.curr_chatID = chatID
            msg = {
                'id': 'MasterBot#%s' % chatID,
                'text': text,
                'episode_done': episode_done,
            }
            local_agent.observe(msg)
            reply = local_agent.act()
            self.last_exchanges[chatID] = [msg, reply]
            if self._is_end_of_conversation(reply['text']) or reply['episode_done']:
                episode_done = True

            if self._is_skip_response(reply['text']):
                print('Skip response from agent for chat #%s' % chatID)
            else:
                print('Send response from agent to chat #%s: %s' % (chatID, reply))
                self._send_message(reply, chatID)
        else:
            print('Message was not recognized as part of any chat. Message skipped.')

        if episode_done:
            self.finished_chats.add(chatID)

    def display(self):
        if self.curr_chatID in self.chats.keys():
            return display_messages(self.last_exchanges[self.curr_chatID])
        else:
            return ''

    def shutdown(self):
        for chatID in self.chats.keys():
            self.chats[chatID].shutdown()
            if chatID not in self.finished_chats:
                self._send_message({'text': '/end'}, chatID)

    def get_chats(self):
        return self.chats.keys()

    def get_finished_chats(self):
        return self.finished_chats
