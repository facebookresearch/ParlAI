#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import errno
import json
import logging
import threading
import time
import websocket
import datetime
from concurrent import futures
import parlai.mturk.core.shared_utils as shared_utils
import parlai.chat_service.core.shared_utils as utils

SOCKET_TIMEOUT = 6


# Socket handler
class ChatServiceMessageSocket:
    """
    ChatServiceMessageSocket is a wrapper around websocket to forward messages from the
    remote server to the ChatServiceManager.
    """

    def __init__(self, server_url, port, message_callback):
        """
        server_url:           url at which the server is to be run
        port:                 port for the socket to operate on
        message_callback:     function to be called on incoming message objects
                              format: message_callback(self, data)
        """
        self.server_url = server_url
        self.port = port
        self.message_callback = message_callback

        self.ws = None
        self.last_pong = None
        self.alive = False

        # initialize the state
        self.listen_thread = None

        # setup the socket
        self.keep_running = True
        self._setup_socket()

    def _safe_send(self, data, force=False):
        if not self.alive and not force:
            # Try to wait a second to send a packet
            timeout = 1
            while timeout > 0 and not self.alive:
                time.sleep(0.1)
                timeout -= 0.1
            if not self.alive:
                # don't try to send a packet if we're still dead
                return False
        try:
            self.ws.send(data)
        except websocket.WebSocketConnectionClosedException:
            # The channel died mid-send, wait for it to come back up
            return False
        return True

    def _ensure_closed(self):
        try:
            self.ws.close()
        except websocket.WebSocketConnectionClosedException:
            pass

    def _send_world_alive(self):
        """
        Registers world with the passthrough server.
        """
        self._safe_send(
            json.dumps(
                {
                    'type': 'world_alive',
                    'content': {'id': 'WORLD_ALIVE', 'sender_id': 'world'},
                }
            ),
            force=True,
        )

    def _setup_socket(self):
        """
        Create socket handlers and registers the socket.
        """

        def on_socket_open(*args):
            shared_utils.print_and_log(logging.DEBUG, 'Socket open: {}'.format(args))
            self._send_world_alive()

        def on_error(ws, error):
            try:
                if error.errno == errno.ECONNREFUSED:
                    self._ensure_closed()
                    self.use_socket = False
                    raise Exception("Socket refused connection, cancelling")
                else:
                    shared_utils.print_and_log(
                        logging.WARN, 'Socket logged error: {}'.format(repr(error))
                    )
            except BaseException:
                if type(error) is websocket.WebSocketConnectionClosedException:
                    return  # Connection closed is noop
                shared_utils.print_and_log(
                    logging.WARN,
                    'Socket logged error: {} Restarting'.format(repr(error)),
                )
                self._ensure_closed()

        def on_disconnect(*args):
            """
            Disconnect event is a no-op for us, as the server reconnects automatically
            on a retry.
            """
            shared_utils.print_and_log(
                logging.INFO, 'World server disconnected: {}'.format(args)
            )
            self.alive = False
            self._ensure_closed()

        def on_message(*args):
            """
            Incoming message handler for messages from the FB user.
            """
            packet_dict = json.loads(args[1])
            if packet_dict['type'] == 'conn_success':
                self.alive = True
                return  # No action for successful connection
            if packet_dict['type'] == 'pong':
                self.last_pong = time.time()
                return  # No further action for pongs
            message_data = packet_dict['content']
            shared_utils.print_and_log(
                logging.DEBUG, 'Message data received: {}'.format(message_data)
            )
            for message_packet in message_data['entry']:
                for message in message_packet['messaging']:
                    self.message_callback(message)

        def run_socket(*args):
            url_base_name = self.server_url.split('https://')[1]
            while self.keep_running:
                try:
                    sock_addr = "wss://{}/".format(url_base_name)
                    self.ws = websocket.WebSocketApp(
                        sock_addr,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_disconnect,
                    )
                    self.ws.on_open = on_socket_open
                    self.ws.run_forever(ping_interval=1, ping_timeout=0.9)
                except Exception as e:
                    shared_utils.print_and_log(
                        logging.WARN,
                        'Socket error {}, attempting restart'.format(repr(e)),
                    )
                time.sleep(0.2)

        # Start listening thread
        self.listen_thread = threading.Thread(
            target=run_socket, name='Main-Socket-Thread'
        )
        self.listen_thread.daemon = True
        self.listen_thread.start()
        time.sleep(1.2)
        while not self.alive:
            try:
                self._send_world_alive()
            except Exception:
                pass
            time.sleep(0.8)


class ChatServiceWorldRunner:
    """
    World Runner.

    Launches worlds, overworlds, etc. Helper for ChatServiceManager.
    """

    def __init__(self, opt, world_path, max_workers, manager, is_debug=False):
        self._world_module = utils.get_world_module(world_path)
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.debug = is_debug
        self._log("Found world module: {}".format(self._world_module))
        opt["is_debug"] = is_debug
        self.manager = manager
        self.system_done = False
        self.opt = opt
        self.tasks = {}  # task ID to task
        self.initialized = False

        def _is_done_initializing(fut):
            e = fut.exception()
            if e is not None:
                self._log('`module_initialize` returned with error {}'.format(repr(e)))
                if self.debug:
                    raise e
            if fut.result():
                print(fut.result())
            if self.debug:
                print("DEBUG: Call to `module_initialize` has completed...")
            self.initialized = True

        if hasattr(self._world_module, "module_initialize"):
            self._log("Initializing world module...")
            # perform any module intialization steps
            init_fn = self._world_module.module_initialize
            self.init_fut = self.executor.submit(init_fn, opt, manager)
            self.init_fut.add_done_callback(_is_done_initializing)
        else:
            self._log("World module does not have `module initialize` function")
            self.initialized = True

    def _log(self, text):
        if self.debug:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("{} DEBUG: {}".format(time, text))

    def is_initialized(self):
        return self.initialized

    def shutdown(self):
        """
        Shutdown the world runner.
        """
        for _, task in self.tasks.items():
            if task.world is not None:
                task.world.shutdown()

        self.system_done = True  # this forces worlds to stop executing parley
        self._log("Executor shutting down.")
        self.executor.shutdown()
        self._log("Shutdown complete.")

    def _run_world(self, task, world_name, agents):
        """
        Run a world until completion.

        :param task:
            TaskState. State of the given task.
        :param world_name:
            string. The name of the world in the module file
        :param agents:
            list. A list of agents that should be in the world.

        :return:
            ret_val: last output of world's parley function. Return None if ERROR
            world_data: data attribute of world, if it has one
        """
        ret_val = None
        world_generator = utils.get_world_fn_attr(
            self._world_module, world_name, "generate_world"
        )
        world = world_generator(self.opt, agents)
        task.world = world

        while not world.episode_done() and not self.system_done:
            ret_val = world.parley()
            time.sleep(0.3)
        world.shutdown()
        world_data = world.data if hasattr(world, "data") else {}
        return ret_val, world_data

    def launch_task_world(self, task_name, world_name, agents):
        """
        Launch a task world.

        Return the job's future.

        :param task_name:
            string. the name of the job thread
        :param world_name:
            string. the name of the task world in the module file
        :param agents:
            list. the list of agents to install in the world

        :return:
            the Futures object corresponding to this launched task
        """
        task = utils.TaskState(task_name, world_name, agents)
        self.tasks[task_name] = task

        def _world_fn():
            utils.print_and_log(logging.INFO, 'Starting task {}...'.format(task_name))
            return self._run_world(task, world_name, agents)

        fut = self.executor.submit(_world_fn)
        task.future = fut
        return fut

    def launch_overworld(self, task_name, overworld_name, onboard_map, overworld_agent):
        """
        Launch an overworld and a subsequent onboarding world.

        Return the job's future

        :param task_name:
            string. the name of the job thread
        :param overworld_name:
            string. the name of the overworld in the module file
        :param onboard_map:
            map. a mapping of overworld return values to the names
            of onboarding worlds in the module file.
        :param overworld_agent:
            The agent to run the overworld with

        :return:
            the Futures object corresponding to running the overworld
        """
        task = utils.TaskState(
            task_name,
            overworld_name,
            [overworld_agent],
            is_overworld=True,
            world_type=None,
        )
        self.tasks[task_name] = task
        agent_state = self.manager.get_agent_state(overworld_agent.id)

        def _world_function():
            world_generator = utils.get_world_fn_attr(
                self._world_module, overworld_name, "generate_world"
            )
            overworld = world_generator(self.opt, [overworld_agent])
            while not overworld.episode_done() and not self.system_done:
                world_type = overworld.parley()
                if world_type is None:
                    time.sleep(0.5)
                    continue

                if world_type == self.manager.EXIT_STR:
                    self.manager._remove_agent(overworld_agent.id)
                    return world_type

                # perform onboarding
                onboard_type = onboard_map.get(world_type)
                if onboard_type:
                    onboard_id = 'onboard-{}-{}'.format(overworld_agent.id, time.time())
                    agent = self.manager._create_agent(onboard_id, overworld_agent.id)
                    agent_state.set_active_agent(agent)
                    agent_state.assign_agent_to_task(agent, onboard_id)
                    _, onboard_data = self._run_world(task, onboard_type, [agent])
                    agent_state.onboard_data = onboard_data
                self.manager.add_agent_to_pool(agent_state, world_type)
                utils.print_and_log(logging.INFO, 'onboarding/overworld complete')

            return world_type

        fut = self.executor.submit(_world_function)
        task.future = fut
        return fut
