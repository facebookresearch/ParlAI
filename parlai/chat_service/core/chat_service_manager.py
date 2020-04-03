#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from asyncio import Future
import copy
import sys
import logging
import datetime
import threading
import time
import traceback
from typing import Dict, Any, Optional, List, Callable

from parlai.chat_service.core.agents import ChatServiceAgent
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
import parlai.chat_service.utils.server as server_utils
from parlai.chat_service.core.world_runner import ChatServiceWorldRunner
from parlai.core.opt import Opt


class AgentState:
    """
    Keep track of Agent State.

    State includes which is the "active" agent - i.e., which agent in which
    world do we message, etc.
    """

    def __init__(self, service_id: int, overworld_agent: ChatServiceAgent):
        self.service_id = service_id
        self.overworld_agent = overworld_agent
        self.active_agent = overworld_agent
        self.task_id_to_agent: Dict[str, ChatServiceAgent] = {}
        self.onboard_data = None
        self.data = {}
        self.stored_data: Dict[str, Any] = {}
        self.time_in_pool: Dict[str, float] = {}

    def get_active_agent(self) -> ChatServiceAgent:
        """
        Return active messenger agent.

        :return:
            a ChatServiceAgent, which corresponds to the active agent for this
            agent state.
        """
        return self.active_agent

    def set_active_agent(self, active_agent: ChatServiceAgent):
        """
        Set active agent for this agent.

        :param active_agent:
            A ChatServiceAgent, the new active agent for this given agent state
        """
        self.active_agent = active_agent

    def get_overworld_agent(self) -> ChatServiceAgent:
        """
        Return overworld messenger agent.

        :return:
            a ChatServiceAgent, which corresponds agent object in the overworld
        """
        return self.overworld_agent

    def get_id(self) -> int:
        """
        Return the agent's ID.

        :return:
            int agent's service ID
        """
        return self.service_id

    def has_task(self, task_id: str) -> bool:
        """
        Determine if an agent is in a task.

        :param task_id:
            task id

        :return:
            if agent is in that task.
        """
        return task_id in self.task_id_to_agent

    def get_agent_for_task(self, task_id: str) -> Optional[ChatServiceAgent]:
        """
        Return ChatServiceAgent for given task id.

        For each "player", a separate agent is created for each task. This
        returns the appropriate MessengerAgent given the task id

        :param task_id:
            task id

        :return:
            ChatServiceAgent object associated with the given task
        """
        if self.has_task(task_id):
            return self.task_id_to_agent[task_id]
        else:
            return None

    def assign_agent_to_task(self, agent: ChatServiceAgent, task_id: str):
        """
        Mark agent in task.

        :param agent:
            ChatServiceAgent object to mark in task
        :param task_id:
            string task name
        """
        self.task_id_to_agent[task_id] = agent


class ChatServiceManager(ABC):
    class ChatServiceMessageSender(ABC):
        """
        ChatServiceMessageSender is a wrapper around requests that simplifies the the
        process of sending content.
        """

        @abstractmethod
        def send_read(self, receiver_id: int):
            """
            Send read receipt to agent at receiver_id.
            """
            pass

        @abstractmethod
        def typing_on(self, receiver_id: int, persona_id: str = None):
            """
            Send typing on msg to agent at receiver_id.
            """
            pass

    EXIT_STR = 'EXIT'

    def __init__(self, opt: Opt):
        """
        Create a ChatServiceManager using the given setup options.
        """
        # Manager attributes
        self.opt = opt
        self.server_url = None
        self.port = 443
        self.agent_pool_change_condition = threading.Condition()
        self.active_worlds: Dict[str, Future] = {}
        self.socket = None
        self.sender = None
        self.running = True
        self.conversation_index = 0
        self.shutting_down = False
        self.bypass_server_setup = self.opt.get('bypass_server_setup')

        # Messaging interaction functions that determine what to do when
        # messages are confirmed as delivered, marked as read by a user, and
        # noted as read by the bot.
        self.confirm_message_delivery = self._confirm_message_delivery
        self.handle_message_read = self._handle_message_read
        self.handle_bot_read = self._handle_bot_read

    def _log_debug(self, text: str):
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_utils.print_and_log(logging.DEBUG, f'{time}: {text}', should_print=True)

    def _parse_config(self, opt: Opt):
        """
        Parse config for task.

        Use this to parse all options and settings necessary to set the variables for
        the conversation
        """
        self.debug = opt['is_debug']
        self.config = opt['config']
        self.overworld = self.config['overworld']
        self.world_path = self.config['world_path']
        self.world_module = utils.get_world_module(self.world_path)
        self.task_configs = self.config['configs']
        self.max_workers = self.config['max_workers']
        self.opt['task'] = self.config['task_name']
        # Deepcopy the opts so the manager opts aren't changed by the world runner
        self.runner_opt = copy.deepcopy(opt)
        self.world_runner = ChatServiceWorldRunner(
            self.runner_opt, self.world_path, self.max_workers, self, opt['is_debug']
        )  # Replace with base runner
        self.max_agents_for = {
            task: cfg.agents_required for task, cfg in self.task_configs.items()
        }
        self.onboard_map = {
            task: cfg.onboarding_name for task, cfg in self.task_configs.items()
        }
        self.taskworld_map = {
            task: cfg.task_name for task, cfg in self.task_configs.items()
        }
        self.service_reference_id = None
        self.parse_additional_args(opt)

    @abstractmethod
    def parse_additional_args(self, opt: Opt):
        """
        Parse any other service specific args here.
        """
        # page id for messenger to be obtained here

    def _get_port(self) -> int:
        """
        Return the port number currently being used.
        """
        return self.port

    def _set_port(self, port_no: int):
        """
        Use a custom port number.

        :param port_no: New port number to be used
        """
        self.port = port_no

    @abstractmethod
    def _complete_setup(self):
        """
        Complete necessary setup items.

        Consider this as a unified method for setting up. Call every other functions
        used in setup from here. To be called during instantiation
        """

    @abstractmethod
    def _load_model(self):
        """
        Load model if necessary.
        """

    def _expire_all_conversations(self):
        """
        Iterate through all sub-worlds and shut them down.
        """
        self.running = False
        for agent_id, overworld_fut in self.agent_id_to_overworld_future.items():
            self.observe_message(
                agent_id,
                'System: The conversation bot going to go offline soon, '
                'finish any active conversations in the next 5 minutes.',
            )
            overworld_fut.cancel()

        # 5 minute grace period for conversations to finish
        time_passed = 0
        while time_passed < 5 * 60:
            any_alive = False
            for task_fut in self.active_worlds.values():
                if task_fut is not None:
                    any_alive = any_alive or task_fut.running()
            if not any_alive:
                break
            time.sleep(1)

        # Tell the worlds that they should be shutting down at this point,
        # agents will refer to this value
        self.shutting_down = True

    def _get_unique_pool(self):
        """
        Return unique pool.

        Returns a filtered version of the agent pool where each agent is
        only listed a maximum of one time.

        :return:
            a dictionary mapping world_types to agent pools
        """
        valid_pools = {}
        for world_type, agent_pool in self.agent_pool.items():
            eligibility_function = utils.get_eligibility_fn(
                self.world_module, world_type
            )
            if eligibility_function is not None:
                valid_pools[world_type] = [
                    w for w in agent_pool if eligibility_function(w)
                ]
            else:
                valid_pools[world_type] = self.agent_pool[world_type]
        return valid_pools

    @abstractmethod
    def restructure_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use this function to restructure the message into the provided format.

        returns the appropriate message.
        """

    @abstractmethod
    def _handle_bot_read(self, agent_id: int):
        """
        Use this function to handle/execute events once the bot has observed the
        message.
        """

    @abstractmethod
    def _confirm_message_delivery(self, event: Dict[str, Any]):
        """
        A callback for when messages are marked as delivered.
        """

    def _handle_message_read(self, event: Dict[str, Any]):
        # If the message was sent by another user (as in during a conversation)
        # then we need to propogate the read back to that user.
        reader = event['sender']['id']
        agent_state = self.get_agent_state(reader)
        if agent_state is None:
            return
        agent = agent_state.get_active_agent()
        if agent is not None:
            for partner in agent.message_partners:
                # We don't know who sent the message that was seen, but we can
                # send a message observed event to everyone else in the chat
                self.sender.send_read(partner.id)  # type: ignore

    def _remove_agent(self, agent_id: int):
        """
        Remove an agent from the system (after they disconnect or leave in some other
        way)
        """
        self.observe_message(agent_id, 'See you later!')
        for world_type in self.agent_pool:
            agent_state = self.get_agent_state(agent_id)
            if agent_state in self.agent_pool[world_type]:
                assert agent_state is not None  # for typing
                self.agent_pool[world_type].remove(agent_state)
                self.remove_agent_from_pool(agent_state, world_type=world_type)
        del self.messenger_agent_states[agent_id]
        del self.agent_id_to_overworld_future[agent_id]

    def _launch_overworld(self, agent_id: int):
        """
        Launch an overworld for the given agent id, replacing the existing overworld if
        it exists already.
        """
        agent_state = self.get_agent_state(agent_id)
        task_id = 'overworld-{}-{}'.format(agent_id, time.time())
        if agent_state is None:
            # new agent
            agent = self._create_agent(task_id, agent_id)
            agent_state = AgentState(agent_id, agent)
            self.messenger_agent_states[agent_id] = agent_state
        else:
            agent = agent_state.overworld_agent
            # kill existing overworld
            self.agent_id_to_overworld_future[agent_id].cancel()

        # launch overworld
        future = self.world_runner.launch_overworld(
            task_id, self.overworld, self.onboard_map, agent
        )

        def _done_callback(fut):
            e = fut.exception()
            if e is not None:
                log_utils.print_and_log(
                    logging.ERROR,
                    'World {} had error {}'.format(task_id, repr(e)),
                    should_print=True,
                )
                traceback.print_exc(file=sys.stdout)
                if self.debug:
                    raise e

        future.add_done_callback(_done_callback)
        self.agent_id_to_overworld_future[agent_id] = future

    def _on_first_message(self, message: Dict[str, Any]):
        """
        Handle first message from player.

        Run when a psid is given that is not paired with any assignment yet.
        Launch an overworld, complete onboarding, etc.

        :param message:
            message sent from agent
        """
        if self.service_reference_id is None:
            self.service_reference_id = message['recipient']['id']

        agent_id = message['sender']['id']
        if self.opt['password'] is not None:
            if message['text'] != self.opt['password']:
                self.observe_message(
                    agent_id,
                    'Sorry, this conversation bot is password-protected. If '
                    'you have the password, please send it now.',
                )
                return

        self._launch_overworld(agent_id)

    def _on_new_message(self, message: Dict[str, Any]):
        """
        Put an incoming message onto the correct agent's message queue.

        :param message:
            message to put on queue
        """
        message = self.restructure_message(message)
        agent_id = message['sender']['id']
        if not self.world_runner.is_initialized():
            self.observe_message(
                agent_id, 'Please wait while the worlds are initializing...'
            )
            self.world_runner.init_fut.result()

        if agent_id not in self.messenger_agent_states:
            self._on_first_message(message)
            return

        agent_state = self.get_agent_state(agent_id)
        assert agent_state is not None
        if agent_state.get_active_agent() is None:
            # return agent to overworld
            if message.get("text", "") and message['text'].upper() == 'EXIT':
                # remove agent from agent_pool
                to_remove = []
                for world_type, _time in agent_state.time_in_pool.items():
                    to_remove.append(world_type)
                for world_type in to_remove:
                    self.remove_agent_from_pool(
                        agent_state, world_type, mark_removed=False
                    )
                self.after_agent_removed(agent_state.get_id())
                agent_state.set_active_agent(agent_state.get_overworld_agent())
            else:
                self.observe_message(
                    agent_id,
                    'Please wait while we pair you with another person. '
                    'If you wish to exit, type *EXIT*.',
                )
                self.sender.typing_on(agent_id)  # type: ignore
        else:
            # If an agent is in a solo world, we can put a typing indicator
            # and mark the message as read
            agent = agent_state.get_active_agent()
            if len(agent.message_partners) == 0:
                self.handle_bot_read(agent.id)  # type: ignore
            agent.put_data(message)

    def add_agent_to_pool(self, agent: AgentState, world_type: str = 'default'):
        """
        Add the agent to pool.

        :param agent:
            AgentState object
        :param world_type:
            Name of world whose pool should now contain agent
        """
        with self.agent_pool_change_condition:
            self._log_debug('Adding agent {} to pool...'.format(agent.service_id))
            # time agent entered agent_pool
            agent.time_in_pool.setdefault(world_type, time.time())
            # add agent to pool
            self.agent_pool.setdefault(world_type, []).append(agent)

    def remove_agent_from_pool(
        self, agent: AgentState, world_type: str = 'default', mark_removed: bool = True
    ):
        """
        Remove agent from the pool.

        :param agent:
            AgentState object
        :param world_type:
            world name
        :param mark_removed:
            whether to mark an agent as removed from the pool
        """
        with self.agent_pool_change_condition:
            self._log_debug('Removing agent {} from pool...'.format(agent.service_id))
            if world_type in self.agent_pool and agent in self.agent_pool[world_type]:
                self.agent_pool[world_type].remove(agent)
                # reset agent's time_in_pool
                if world_type in agent.time_in_pool:
                    del agent.time_in_pool[world_type]
                # maybe mark agent as removed
                if mark_removed:
                    agent.stored_data['removed_from_pool'] = True
                    if self.service_reference_id is not None:
                        self.mark_removed(agent.service_id, self.service_reference_id)

    @abstractmethod
    def _create_agent(self, task_id: str, agent_id: int) -> ChatServiceAgent:
        """
        Initialize an agent and return it.

        Called each time an agent is placed into a new task.

        :param task_id:
            task identifier
        :param agent_id:
            agent id
        """

    def _get_agent(self, agent_id: int, task_id: str) -> Optional[ChatServiceAgent]:
        """
        Return agent object for given agent ID and task ID.

        :param agent_id:
            agent identifier
        :param task_id:
            task name

        :return:
            ChatServiceAgent object associated with given agent ID and task ID if
            possible, else None
        """
        agent_state = self.get_agent_state(agent_id)
        if agent_state is not None:
            if agent_state.has_task(task_id):
                return agent_state.get_agent_for_task(task_id)
        return None

    def get_agent_state(self, agent_id: int) -> Optional[AgentState]:
        """
        Return agent state.

        :param agent_id:
            agent identifier

        :return:
            AgentState object if agent_id is being tracked, else None
        """
        if agent_id in self.messenger_agent_states:
            return self.messenger_agent_states[agent_id]
        return None

    @abstractmethod
    def setup_server(self):
        """
        Prepare the Chat Service server for handling messages.
        """
        pass

    @abstractmethod
    def setup_socket(self):
        """
        Set up socket to start communicating to workers.
        """
        pass

    def init_new_state(self):
        """
        Prepare for new run.

        Initialize everything in the agent, task, and thread states
        """
        self.agent_pool = {}
        self.messenger_agent_states = {}
        self.task_to_agent_ids = {}
        self.agent_id_to_overworld_future = {}

    def start_new_run(self):
        """
        Begin new run.
        """
        self.run_id = str(int(time.time()))
        self.task_group_id = '{}_{}'.format(self.opt['task'], self.run_id)

    def check_timeout_in_pool(
        self,
        world_type: str,
        agent_pool: List[AgentState],
        max_time_in_pool: int,
        backup_task: str = None,
    ):
        """
        Check for timed-out agents in pool.

        :param world_type:
            world type
        :param agent_pool:
            list of AgentStates
        :param max_time_in_pool:
            maximum time allowed for agent to be in pool
        :param backup_task:
            backup_task to start if we reach a timeout in the original pool
        """
        for agent_state in agent_pool:
            time_in_pool = agent_state.time_in_pool.get(world_type)
            if time_in_pool and time.time() - time_in_pool > max_time_in_pool:
                # remove agent from agent_pool
                self.remove_agent_from_pool(agent_state, world_type)
                # put agent back in overworld
                agent_state.set_active_agent(agent_state.get_overworld_agent())

                agent_state.stored_data['removed_after_timeout'] = True
                self.after_agent_removed(agent_state.service_id)

                if backup_task is not None:
                    self.add_agent_to_pool(agent_state, backup_task)

                # reset wait message state
                agent_state.stored_data['seen_wait_message'] = False

            elif time_in_pool and time.time() - time_in_pool > 30:
                # tell agent that a match is taking longer than
                # expected
                if (
                    not agent_state.stored_data.get('seen_wait_message')
                    or not agent_state.stored_data['seen_wait_message']
                ):
                    self.observe_message(
                        agent_state.service_id,
                        'Pairing is taking longer than expected. '
                        'If you wish to exit, type *EXIT*.',
                    )
                    self.sender.typing_on(agent_state.service_id)  # type: ignore
                    agent_state.stored_data['seen_wait_message'] = True

    def _get_done_callback_for_agents(
        self, task_id: str, world_type: str, agents: List[ChatServiceAgent]
    ) -> Callable[[Future], None]:
        """
        Create done callback for finishing task world with particular agents.

        :param task_id:
            task identifier
        :param world_type:
            world name
        :param agents:
            agents for which we are retrieving done callback

        :return:
            the done callback, i.e. the callback function for when agents are done
            in a world.
        """

        def _done_callback(fut):
            """
            Log and raise exception of task world, if there is one.

            Additionally, set active agent to overworld agent.
            """
            e = fut.exception()
            if e is not None:
                log_utils.print_and_log(
                    logging.ERROR,
                    'World {} had error {}'.format(world_type, repr(e)),
                    should_print=True,
                )
                traceback.print_exc(file=sys.stdout)
                for agent in agents:
                    self.observe_message(
                        agent.id, 'Sorry, this world closed. Returning to overworld.'
                    )
            else:
                log_utils.print_and_log(
                    logging.INFO,
                    'World {} had no error'.format(world_type),
                    should_print=True,
                )
            self.active_worlds[task_id] = None
            for agent in agents:
                self.after_agent_removed(agent.id)
                agent_state = self.get_agent_state(agent.id)
                agent_state.data = agent.data
                next_task = agent.data.get("next_task")
                log_utils.print_and_log(logging.INFO, "Next task: {}".format(next_task))
                if next_task is None:
                    self._launch_overworld(agent.id)
                    agent_state.set_active_agent(agent_state.get_overworld_agent())
                elif next_task == self.EXIT_STR:
                    self._remove_agent(agent.id)
                else:
                    self.add_agent_to_pool(agent_state, next_task)

        return _done_callback

    def start_task(self):
        """
        Begin handling task.

        Periodically check to see when enough agents are in the agent pool to start an
        instance of the task. Continue doing this until the desired number of
        conversations is had.
        """

        self.running = True
        while self.running:
            # Loop forever until the server is shut down
            with self.agent_pool_change_condition:
                valid_pools = self._get_unique_pool()
                for world_type, agent_pool in valid_pools.items():
                    # check if agent has exceeded max time in pool
                    world_config = self.task_configs[world_type]
                    if world_config.max_time_in_pool is not None:
                        self.check_timeout_in_pool(
                            world_type,
                            agent_pool,
                            world_config.max_time_in_pool,
                            world_config.backup_task,
                        )

                    needed_agents = self.max_agents_for[world_type]
                    if len(agent_pool) >= needed_agents:
                        log_utils.print_and_log(
                            logging.INFO, 'starting pool', should_print=True
                        )
                        # enough agents in pool to start new conversation
                        self.conversation_index += 1
                        task_id = 't_{}'.format(self.conversation_index)

                        # Add the required number of valid agents to the conv
                        agent_states = [w for w in agent_pool[:needed_agents]]
                        agents = []
                        for state in agent_states:
                            agent = self._create_agent(task_id, state.get_id())
                            agent.onboard_data = state.onboard_data
                            agent.data = state.data
                            state.assign_agent_to_task(agent, task_id)
                            state.set_active_agent(agent)
                            agents.append(agent)
                            # reset wait message state
                            state.stored_data['seen_wait_message'] = False
                        assign_role_function = utils.get_assign_roles_fn(
                            self.world_module, self.taskworld_map[world_type]
                        )
                        if assign_role_function is None:
                            assign_role_function = utils.default_assign_roles_fn
                        assign_role_function(agents)
                        # Allow task creator to filter out workers and run
                        # versions of the task that require fewer agents
                        agents = [a for a in agents if a.disp_id is not None]
                        for a in agents:
                            # Remove selected workers from the agent pool
                            self.remove_agent_from_pool(
                                self.get_agent_state(a.id),
                                world_type=world_type,
                                mark_removed=False,
                            )
                        for a in agents:
                            partner_list = agents.copy()
                            partner_list.remove(a)
                            a.message_partners = partner_list

                        done_callback = self._get_done_callback_for_agents(
                            task_id, world_type, agents
                        )

                        # launch task world.
                        future = self.world_runner.launch_task_world(
                            task_id, self.taskworld_map[world_type], agents
                        )
                        future.add_done_callback(done_callback)
                        self.active_worlds[task_id] = future

            time.sleep(utils.THREAD_MEDIUM_SLEEP)

    def shutdown(self):
        """
        Handle any client shutdown cleanup.
        """
        try:
            self.is_running = False
            self.world_runner.shutdown()
            if not self.bypass_server_setup:
                self.socket.keep_running = False
            self._expire_all_conversations()
        except BaseException as e:
            log_utils.print_and_log(logging.ERROR, f'world ended in error: {e}')

        finally:
            if not self.bypass_server_setup:
                server_utils.delete_server(self.server_task_name, self.opt['local'])

    @abstractmethod
    def observe_message(
        self,
        receiver_id: int,
        text: str,
        quick_replies: List[str] = None,
        persona_id: str = None,
    ):
        """
        Send a message through the message manager.

        :param receiver_id:
            identifier for agent to send message to
        :param text:
            text to send
        :param quick_replies:
            list of quick replies
        :param persona_id:
            identifier of persona
        """

    # Other util functions

    def _handle_webhook_event(self, event: Dict[str, Any]):
        """
        Use this if the service uses webhooks.
        """
        pass

    def mark_removed(self, agent_id: int, pageid: int):
        """
        Mark the agent as removed from the pool.

        Can be overriden to change other metadata linked to agent removal.

        :param agent_id:
            int agent psid
        :param pageid:
            int page id
        """
        pass

    def after_agent_removed(self, agent_id: int):
        """
        Perform any changes to metadata on agent removal.

        override if extra bookkeeping must be done when removing agent
        """
        pass

    # Agent Interaction Functions [Also extra utils]

    def observe_payload(
        self,
        receiver_id: str,
        data: Dict[Any, Any],
        quick_replies: List[str] = None,
        persona_id: str = None,
    ):
        """
        Send a payload through the message manager.

        :param receiver_id:
            int identifier for agent to send message to
        :param data:
            object data to send
        :param quick_replies:
            list of quick replies
        :param persona_id:
            identifier of persona
        """
        pass

    def upload_attachment(self, payload: Dict[str, str]) -> str:
        """
        Upload an attachment and return an attachment ID.

        :param payload:
            dict with the following format:
                {'type': <TYPE>, 'url': <URL>} or
                {'type': <TYPE>, 'filename': <FILENAME>, 'format': <FILEFORMAT>}.
                For example,
                {'type': 'image', 'filename': 'test.png', 'format': 'png'}

        :return:
            attachment id associated with attachment.
        """
        pass
