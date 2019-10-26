#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class AgentState:
    """Keep track of Agent State.

    State includes which is the "active" agent - i.e., which agent in which
    world do we message, etc.
    """

    def __init__(self, messenger_id, overworld_agent):
        self.messenger_id = messenger_id
        self.overworld_agent = overworld_agent
        self.active_agent = overworld_agent
        self.task_id_to_agent = {}
        self.onboard_data = None
        self.stored_data = {}
        self.time_in_pool = {}

    def get_active_agent(self):
        """Return active messenger agent.

        :return:
            a MessengerAgent, which corresponds to the active agent for this
            agent state.
        """
        return self.active_agent

    def set_active_agent(self, active_agent):
        """Set active agent for this agent.

        :param active_agent:
            A MessengerAgent, the new active agent for this given agent state

        """
        self.active_agent = active_agent

    def get_overworld_agent(self):
        """Return overworld messenger agent.

        :return:
            a MessengerAgent, which corresponds agent object in the overworld
        """
        return self.overworld_agent

    def get_id(self):
        """Return agent's ID.

        :return:
            int agent ID
        """
        return self.messenger_id

    def has_task(self, task_id):
        """Determine if an agent is in a task.

        :param task_id:
            string task id

        :return:
            if agent is in that task.
        """
        return task_id in self.task_id_to_agent

    def get_agent_for_task(self, task_id):
        """Return MessengerAgent for given task id.

        For each "player", a separate agent is created for each task. This
        returns the appropriate MessengerAgent given the task id

        :param task_id:
            string, task id

        :return:
            messenger agent object associated with the given task
        """
        if self.has_task(task_id):
            return self.task_id_to_agent[task_id]
        else:
            return None

    def assign_agent_to_task(self, agent, task_id):
        """Mark agent in task.

        :param agent:
            MessengerAgent object to mark in task
        :param task_id:
            string task name
        """
        self.task_id_to_agent[task_id] = agent


class ChatServiceManager:
    def __init__(self, opt):
        """Create a ChatServiceManager using the given setup options
        """
        # # Manager attributes
        # self.opt = opt
        # self.server_url = None
        # self.port = 443
        # self.agent_pool_change_condition = threading.Condition()
        # self.active_worlds = {}
        # self.socket = None
        # self.sender = None
        # self.running = True
        # self.conversation_index = 0
        # self.shutting_down = False
        # self.bypass_server_setup = self.opt.get('bypass_server_setup')
        #
        # # Messaging interaction functions that determine what to do when
        # # messages are confirmed as delivered, marked as read by a user, and
        # # noted as read by the bot.
        # self.confirm_message_delivery = self._confirm_message_delivery
        # self.handle_message_read = self._handle_message_read
        # self.handle_bot_read = self._handle_bot_read
        pass

    def _complete_setup(self):
        """Complete necessary setup items."""
        pass

    def _load_model(self):
        """Load model if necessary."""
        pass

    def _expire_all_conversations(self):
        """Iterate through all sub-worlds and shut them down."""
        pass

    def _get_unique_pool(self):
        """Return unique pool.

        Returns a filtered version of the agent pool where each agent is
        only listed a maximum of one time.

        :return:
            a dictionary mapping world_types to agent pools
        """
        pass

    def _handle_bot_read(self, agent_id):
        pass

    def _confirm_message_delivery(self, event):
        pass

    def _handle_message_read(self, event):
        # If the message was sent by another user (as in during a conversation)
        # then we need to propogate the read back to that user.
        pass

    def _on_first_message(self, message):
        """Handle first message from player.

        Run when a psid is given that is not paired with any assignment yet.
        Launch an overworld, complete onboarding, etc.

        :param message:
            message sent from agent
        """
        pass

    def _handle_webhook_event(self, event):
        pass

    def _on_new_message(self, message):
        """Put an incoming message onto the correct agent's message queue.

        :param message:
            message to put on queue
        """

    def add_agent_to_pool(self, agent, world_type='default'):
        """Add the agent to pool.

        :param agent:
            MessengerAgent object
        :param world_type:
            Name of world whose pool should now contain agent
        """
        pass

    def mark_removed(self, agent_id, pageid):
        """Mark the agent as removed from the pool.

        Can be overriden to change other metadata linked to agent removal.

        :param agent_id:
            int agent psid
        :param pageid:
            int page id
        """
        pass

    def remove_agent_from_pool(self, agent, world_type='default', mark_removed=True):
        """Remove agent from the pool.

        :param agent:
            Agent object
        :param world_type:
            string, world name
        :param mark_removed:
            bool, whether to mark an agent as removed from the pool
        """
        pass

    def after_agent_removed(self, agent_id):
        """Perform any changes to metadata on agent removal.

        override if extra bookkeeping must be done when removing agent
        """
        pass

    def _create_agent(self, task_id, agent_id):
        """Initialize an agent and return it.

        Called each time an agent is placed into a new task.

        :param task_id:
            string task identifier
        :param agent_id:
            int agent id
        """
        pass

    def _get_agent(self, agent_id, task_id):
        """Return agent object for given agent ID and task ID.

        :param agent_id:
            int agent identifier
        :param task_id:
            string task name

        :return:
            MessengerAgent object associated with given agent ID and task ID if
            possible, else None
        """
        pass

    def get_agent_state(self, agent_id):
        """Return agent state.

        :param agent_id:
            int agent identifier

        :return:
            AgentState object if agent_id is being tracked, else None
        """
        pass

    def setup_server(self):
        """Prepare the Chat Service server for handling messages."""
        pass

    def get_app_token(self):
        """Find and return an app access token."""
        pass

    def setup_socket(self):
        """Set up socket to start communicating to workers."""
        pass

    def init_new_state(self):
        """Prepare for new run.

        Initialize everything in the agent, task, and thread states
        """
        pass

    def start_new_run(self):
        """Begin new run."""
        pass

    def check_timeout_in_pool(self, world_type, agent_pool, max_time_in_pool):
        """Check for timed-out agents in pool.

        :param world_type:
            string world type
        :param agent_pool:
            list of ``AgentState``s
        :param max_time_in_pool:
            int maximum time allowed for agent to be in pool
        """
        pass

    def start_task(self):
        """Begin handling task.

        Periodically check to see when enough agents are in the agent pool
        to start an instance of the task. Continue doing this until the desired
        number of conversations is had.
        """
        pass

    def shutdown(self):
        """Handle any client shutdown cleanup."""
        pass

    # Agent Interaction Functions #

    def observe_message(self, receiver_id, text, quick_replies=None, persona_id=None):
        """Send a message through the message manager.

        :param receiver_id:
            int identifier for agent to send message to
        :param text:
            string text to send
        :param quick_replies:
            list of quick replies
        :param persona_id:
            identifier of persona
        """
        pass

    def observe_payload(self, receiver_id, data, quick_replies=None, persona_id=None):
        """Send a payload through the message manager.

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

    def upload_attachment(self, payload):
        """Upload an attachment and return an attachment ID.

        :param payload:
            dict with the following format:
                {'type': <TYPE>, 'url': <URL>} or
                {'type': <TYPE>, 'filename': <FILENAME>, 'format': <FILEFORMAT>}.
                For example,
                {'type': 'image', 'filename': 'test.png', 'format': 'png'}
        """
        pass
