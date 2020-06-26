#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import time
import datetime
from concurrent import futures
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils


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
            log_utils.print_and_log(
                logging.INFO, 'Starting task {}...'.format(task_name)
            )
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
                    agent.data = overworld_agent.data
                    agent_state.set_active_agent(agent)
                    agent_state.assign_agent_to_task(agent, onboard_id)
                    _, onboard_data = self._run_world(task, onboard_type, [agent])
                    agent_state.onboard_data = onboard_data
                    agent_state.data = agent.data
                self.manager.add_agent_to_pool(agent_state, world_type)
                log_utils.print_and_log(logging.INFO, 'onboarding/overworld complete')

            return world_type

        fut = self.executor.submit(_world_function)
        task.future = fut
        return fut
