"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from parlai.core.worlds import World, DialogPartnerWorld
from parlai.core.agents import Agent, create_agent_from_shared

import requests
import os
import json
import time


class ConvAIWorld(World):
    """
    ConvAIWorld provides conversations with participants in the convai.io competition.
    It creates a new DialogPartnerWorld for each new conversation.
    Each DialogPartnerWorld populated with two new instances of agents: ConvAIAgent and yours local agent.
    Information about yours agent should be provided via 'shared' parameter. Agents from 'agents' parameter are ignored.
    """

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt)

        if shared is None:
            raise RuntimeError("Agents should be provided via 'shared' parameter")

        self.shared = shared
        # Current chat id
        self.chat = None
        # All active and finished (but not cleared yet) chats
        self.chats = {}
        # Finished chats
        self.finished_chats = set()
        # Pool of messages from RouterBot
        self.messages = []
        # Url of RouterBot
        self.router_bot_url = opt.get('router_bot_url')
        # Delay before new request to RouterBot: minimum 1 sec
        self.router_bot_pull_delay = int(opt.get('router_bot_pull_delay'))
        if self.router_bot_pull_delay < 1:
            self.router_bot_pull_delay = 1
        # Id of local bot used to communicate with RouterBot
        self.bot_id = opt.get('bot_id')
        # The maximum number of open dialogs. Use -1 for unlimited number of open dialogs
        self.bot_capacity = int(opt.get('bot_capacity'))
        # RouterBot url with current bot id
        self.bot_url = os.path.join(self.router_bot_url, self.bot_id)

    def _get_updates(self):
        """
        Make HTTP request to Router Bot for new messages
        :return: list of new messages received since last request
        """
        res = requests.get(os.path.join(self.bot_url, 'getUpdates'))
        if res.status_code != 200:
            print(res.text)
            res.raise_for_status()
        return res.json()

    def _send_message(self, observation, chat):
        """
        Make HTTP request to Router Bot to post new message
        :param observation: message that will be sent to server
        :param chat: id of chat
        :return: None
        """
        if self._is_end_of_conversation(observation['text']):
            data = {
                'text': '/end',
                'evaluation': {
                    'quality': 0,
                    'breadth': 0,
                    'engagement': 0
                }
            }
        else:
            data = {
                'text': observation['text'],
                'evaluation': 0
            }
        message = {
            'chat_id': chat,
            'text': json.dumps(data)
        }

        headers = {
            'Content-Type': 'application/json'
        }

        res = requests.post(os.path.join(self.bot_url, 'sendMessage'), json=message, headers=headers)
        if res.status_code != 200:
            print(res.text)
            res.raise_for_status()

    @staticmethod
    def _is_begin_of_conversation(message):
        return message.startswith('/start ')

    @staticmethod
    def _is_end_of_conversation(message):
        return message == '/end'

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
        return message.replace('/start ', '')

    def _init_chat(self, chat):
        """
        Create new chat for new dialog. Chat consists of new instance of ConvAIAgent,
        new instance of your own Agent and new instance DialogPartnerWorld where this two agents
        will communicate. Information about class of your local agent getting from shared data.
        :param chat: chat id
        :return: tuple with instances of ConvAIAgent, your local agent, DialogPartnerWorld
        """
        agent_info = self.shared["agents"][0]
        
        # Add refs to current world instance and chat id to agent 'opt' parameter
        if 'opt' not in agent_info.keys() or agent_info['opt'] is None:
            agent_info['opt'] = {}
        agent_info['opt']['convai_world'] = self
        agent_info['opt']['convai_chat'] = chat

        local_agent = create_agent_from_shared(agent_info)
        remote_agent = ConvAIAgent({'chat': chat})
        world = DialogPartnerWorld({'task': 'ConvAI Dialog'}, [remote_agent, local_agent])
        self.chats[chat] = (remote_agent, local_agent, world)
        return self.chats[chat]

    def cleanup_finished_chat(self, chat):
        """
        Shutdown specified chat and remove it from lists
        :param chat: chat id
        :return: None
        """
        if chat in self.finished_chats:
            self.chats.pop(chat, None)[2].shutdown()
            self.finished_chats.remove(chat)

    def cleanup_finished_chats(self):
        """
        Shutdown all finished chats and remove them from lists
        :return: None
        """
        for chat in self.finished_chats.copy():
            self.cleanup_finished_chat(chat)

    def pull_new_messages(self):
        """
        Requests server for new messages and processes every message.
        If message starts with '/start' then will create new chat and adds message to stack.
        If message has same id as already existing chat then will add to message stack.
        Other messages will be ignored.
        If after processing all messages message stack is still empty then new request to server will be performed.
        :return: None
        """
        print("Wait for new messages from server", end="", flush=True)
        while True:
            time.sleep(self.router_bot_pull_delay)
            print(".", end="", flush=True)
            msgs = self._get_updates()
            if len(msgs) > 0:
                for msg in msgs:
                    print("\nProceed message: %s" % msg)
                    text = self._get_message_text(msg)
                    chat = self._get_chat_id(msg)

                    if self.chats.get(chat, None) is not None:
                        print("Message was recognized as part of chat #%s" % chat)
                        self.messages.append((chat, text))
                    elif self._is_begin_of_conversation(text):
                        print("Message was recognised as start of new chat #%s" % chat)
                        if self.bot_capacity == -1 or 0 <= self.bot_capacity > (len(self.chats) - len(self.finished_chats)):
                            self._init_chat(chat)
                            text = self._strip_start_message(text)
                            self.messages.append((chat, text))
                            print("New world and agents for chat #%s created." % chat)
                        else:
                            print("Can't start new chat #%s due to bot capacity limit reached." % chat)
                    else:
                        print("Message wasn't recognized as part of any chat. Message skipped.")
                if len(self.messages) > 0:
                    break
                else:
                    print("Wait for new messages from server", end="", flush=True)

    def parley(self):
        """
        Pops next message from stack, gets corresponding chat, agents, world and performs communication between agents.
        Result of communication will be send to server.
        If message stack is empty then server will be requested for new messages.
        :return: None
        """
        print("Try to cleanup finished chat before new parley.")
        self.cleanup_finished_chats()

        if len(self.messages) == 0:
            print("Message stack is empty. Try to request new messages from server.")
            self.pull_new_messages()

        print("Pop next message from stack")

        (chat, text) = self.messages.pop(0)
        episode_done = self._is_end_of_conversation(text)
        (remote_agent, local_agent, world) = self.chats.get(chat, (None, None, None))

        if remote_agent is not None and local_agent is not None and world is not None:
            self.chat = chat
            remote_agent.text = text
            remote_agent.episode_done = episode_done
            print("Parley:")
            world.parley()
            observation = remote_agent.observation
            if self._is_end_of_conversation(observation['text']) or observation['episode_done']:
                episode_done = True

            if self._is_skip_response(observation['text']):
                print("Skip response from agent for chat #%s" % chat)
            else:
                print("Send response from agent to chat #%s: %s" % (chat, observation))
                self._send_message(observation, chat)
        else:
            print("Message wasn't recognized as part of any chat. Message skipped.")

        if episode_done:
            self.finished_chats.add(chat)

    def display(self):
        if self.chat in self.chats.keys():
            return self.chats[self.chat][2].display()
        else:
            return ''

    def shutdown(self):
        for chat in self.chats.keys():
            self.chats[chat][2].shutdown()
            if chat not in self.finished_chats:
                self._send_message({'text': '/end'}, chat)

    def get_chats(self):
        return self.chats.keys()

    def get_finished_chats(self):
        return self.finished_chats

    def get_world(self, chat):
        return self.chats[chat][2]


class ConvAIAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = "MasterBot#%s" % opt['chat']
        self.text = None
        self.observation = None
        self.episode_done = False

    def act(self):
        return {
            'id': self.id,
            'text': self.text,
            'episode_done': self.episode_done
        }
